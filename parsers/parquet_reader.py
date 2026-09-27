from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from .dataset import Frame, Dataset, KBAR_PER_EV_PER_ANG3


# ──────────────────────────────────────────────
# Expected column schema (produced by scripts/vasprun2data.py)
# ──────────────────────────────────────────────
#
# The parquet is stored in *long* format: one row per (Step, Atom_Index).
# Per-frame scalars (energy, lattice) are repeated on every atom row of that
# step — we read them off the first row of each Step group.

_POS_COLS     = ["X", "Y", "Z"]                          # Cartesian Å
_FORCE_COLS   = ["Force_X", "Force_Y", "Force_Z"]        # eV/Å
# Lattice vectors flattened row-major: row a = (ax, ay, az), etc.
_LATTICE_COLS = [
    "Lattice_ax", "Lattice_ay", "Lattice_az",
    "Lattice_bx", "Lattice_by", "Lattice_bz",
    "Lattice_cx", "Lattice_cy", "Lattice_cz",
]
# Optional virial stress in Voigt order (xx, yy, zz, yz, xz, xy), stored in
# the file as kBar with VASP's sign (straight from vasprun.xml via
# scripts/vasprun2data.py); converted to eV/Å³ ASE convention on read.
_STRESS_COLS  = [
    "Stress_xx", "Stress_yy", "Stress_zz",
    "Stress_yz", "Stress_xz", "Stress_xy",
]

_REQUIRED = ["Step", "Atom_Index", "Element", "Total_Energy"] + _POS_COLS + _FORCE_COLS


def _voigt_to_tensor(voigt: np.ndarray) -> np.ndarray:
    """Expand a Voigt (6,) stress vector to a symmetric (3, 3) tensor, matching
    the convention used by extxyz_reader / vasp_reader."""
    xx, yy, zz, yz, xz, xy = voigt
    return np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ])


def read_parquet(
    filepath:    str,
    frame_start: int = 0,
    frame_end:   int = -1,
    stride:      int = 1,
    logger:      logging.Logger | None = None,
) -> Dataset:
    """
    Read an AIMD trajectory stored as a long-format parquet and return a Dataset.

    Expected columns (see ``scripts/vasprun2data.py``):
        Step, Atom_Index, Element,
        X, Y, Z                     -- Cartesian positions in Å
        Force_X, Force_Y, Force_Z   -- forces in eV/Å
        Total_Energy                -- total energy of the step in eV (repeated)
        Lattice_ax .. Lattice_cz    -- the 3x3 cell, row-major, repeated per row
        Stress_xx .. Stress_xy      -- optional Voigt virial stress

    Positions are converted to fractional coordinates to match the convention
    used by ``vasp_reader`` / ``potentials``. Frame selection (frame_start /
    frame_end / stride) is positional over the sorted unique ``Step`` values,
    identical to ``vasp_reader``'s indexing into ``ionic_steps``.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"parquet file not found: '{filepath}'.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    # Local import: pandas/pyarrow are only needed for this reader, so importing
    # at module scope would make the whole parsers package depend on them.
    import pandas as pd

    if logger:
        logger.info(f"Parsing {path} with pandas ...")
    t0 = time.time()
    df = pd.read_parquet(path)

    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"parquet '{filepath}' is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}."
        )

    if not all(c in df.columns for c in _LATTICE_COLS):
        raise ValueError(
            f"parquet '{filepath}' has no lattice columns ({_LATTICE_COLS}). "
            f"BVFF needs a cell for every frame (Ewald, BV neighbour search). "
            f"Regenerate the parquet with cell information using "
            f"scripts/vasprun2data.py (which now writes the Lattice_* columns)."
        )

    has_stress = all(c in df.columns for c in _STRESS_COLS)

    # Positional frame selection over the ordered unique steps.
    all_steps = sorted(df["Step"].unique().tolist())
    n_total   = len(all_steps)

    start = frame_start
    end   = n_total if frame_end == -1 else min(frame_end + 1, n_total)
    if start >= n_total:
        raise ValueError(
            f"frame_start ({start}) exceeds total number of frames ({n_total})."
        )

    sel_positions = list(range(start, end, stride))
    sel_steps     = [all_steps[i] for i in sel_positions]
    n_sel         = len(sel_steps)

    if logger:
        logger.info(
            f"Extracting {n_sel} frames "
            f"(start={start}, end={end - 1}, stride={stride}) ..."
        )

    # Restrict to selected steps before grouping — cheaper than grouping the
    # whole table when only a subsample is kept.
    sub = df[df["Step"].isin(sel_steps)]
    groups = {step: g for step, g in sub.groupby("Step", sort=True)}

    if logger:
        from src.outputs import progress_iter
        step_iter = progress_iter(list(enumerate(sel_steps)), label="extract frames")
    else:
        step_iter = list(enumerate(sel_steps))

    frames: list[Frame] = []
    ref_species: list[str] | None = None
    ref_unique:  list[str] | None = None

    for pos_i, step in step_iter:
        g = groups[step].sort_values("Atom_Index")

        species = [str(s) for s in g["Element"].tolist()]
        unique  = list(dict.fromkeys(species))

        if ref_species is None:
            ref_species = species
            ref_unique  = unique
        elif species != ref_species:
            raise ValueError(
                f"parquet step {step} has species order different from the first "
                f"selected frame. All frames in one file must share atom order."
            )

        cell = g.iloc[0][_LATTICE_COLS].to_numpy(dtype=float).reshape(3, 3)
        if np.linalg.det(cell) == 0:
            raise ValueError(
                f"parquet step {step} has a singular cell {cell.tolist()}. "
                f"Check the Lattice_* columns."
            )

        cart_positions = g[_POS_COLS].to_numpy(dtype=float)        # (N, 3) Å
        frac_positions = cart_positions @ np.linalg.inv(cell)

        stress = None
        if has_stress:
            # File stores VASP-convention kBar; normalize to the Frame
            # convention (eV/Å³, ASE sign), same as vasp_reader.
            stress = -_voigt_to_tensor(
                g.iloc[0][_STRESS_COLS].to_numpy(dtype=float)
            ) / KBAR_PER_EV_PER_ANG3

        frames.append(Frame(
            index     = sel_positions[pos_i],
            lattice   = cell,
            species   = species,
            positions = frac_positions,
            energy    = float(g.iloc[0]["Total_Energy"]),
            forces    = g[_FORCE_COLS].to_numpy(dtype=float),
            stress    = stress,
            source    = str(path),
        ))

    if not frames:
        raise ValueError(
            f"No frames selected from '{filepath}' "
            f"(start={frame_start}, end={frame_end}, stride={stride})."
        )

    if logger:
        logger.info(
            f"Extracted {len(frames)} frames from {path} in {time.time() - t0:.1f}s"
        )

    return Dataset(
        frames  = frames,
        n_atoms = len(ref_species),
        species = ref_unique,
    )


if __name__ == "__main__":
    import sys
    data = read_parquet(sys.argv[1] if len(sys.argv) > 1 else "data.parquet")
    print(f"Total frames loaded : {len(data.frames)}")
    print(f"Number of atoms     : {data.n_atoms}")
    print(f"Species             : {data.species}")
