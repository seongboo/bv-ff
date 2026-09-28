from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.outputs import Vasprun

from .dataset import Frame, Dataset, KBAR_PER_EV_PER_ANG3


def _extract_stress(ionic_step: dict) -> np.ndarray | None:
    stress = ionic_step.get("stress", None)
    if stress is None:
        return None
    # vasprun.xml reports kBar with VASP's sign; normalize to the Frame
    # convention (eV/Å³, ASE sign σ = (1/V) ∂E/∂ε), same flip ASE's vasp
    # reader applies.
    return -np.array(stress) / KBAR_PER_EV_PER_ANG3


def read_vasprun(
    filepath:    str,
    frame_start: int = 0,
    frame_end:   int = -1,
    stride:      int = 1,
    logger:      logging.Logger | None = None,
    energy_key:  str = "e_fr_energy",
) -> Dataset:
    """
    Read vasprun.xml and return a Dataset containing selected frames.

    Args:
        filepath:    Path to vasprun.xml
        frame_start: First frame index to read (inclusive)
        frame_end:   Last frame index to read (inclusive), -1 means all frames
        stride:      Step between consecutive sampled frames (>= 1)
        logger:      Optional logger; when provided, parse/extract progress is reported.
        energy_key:  Which per-step energy to use as the fit reference:
                     "e_fr_energy" (default) is TOTEN, the electronic free
                     energy including the smearing entropy term;
                     "e_0_energy" is energy(sigma→0), the entropy-extrapolated
                     total usually preferred for force-field fitting. The
                     per-ref_group offset removal in the loss absorbs the
                     roughly-constant part of the difference either way, so
                     the default is kept for backward compatibility.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"vasprun.xml not found: '{filepath}'.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    if logger:
        # pymatgen parses in one opaque call (no progress hook): give the file
        # size so a long silence can be judged against it.
        size_mb = Path(path).stat().st_size / 1e6
        logger.info(f"Parsing {path} ({size_mb:.0f} MB) with pymatgen ...")
    t0 = time.time()
    vasprun = Vasprun(
        str(path),
        parse_dos=False,
        parse_eigen=False,
        parse_projected_eigen=False,
    )
    if logger:
        logger.info(f"pymatgen parse done in {time.time() - t0:.1f}s")

    ionic_steps = vasprun.ionic_steps
    n_total     = len(ionic_steps)

    start = frame_start
    end   = n_total if frame_end == -1 else min(frame_end + 1, n_total)

    if start >= n_total:
        raise ValueError(
            f"frame_start ({start}) exceeds total number of frames ({n_total})."
        )

    selected_steps = ionic_steps[start:end:stride]
    selected_idx   = list(range(start, end, stride))
    n_sel          = len(selected_steps)

    all_species    = [str(s) for s in vasprun.atomic_symbols]
    unique_species = list(dict.fromkeys(all_species))

    if logger:
        logger.info(
            f"Extracting {n_sel} frames "
            f"(start={start}, end={end - 1}, stride={stride}) ..."
        )

    if logger:
        from bvff.core.outputs import progress_iter
        step_iter = progress_iter(selected_steps, label="extract frames")
    else:
        step_iter = selected_steps

    frames: list[Frame] = []
    t_ext = time.time()
    for i, step in enumerate(step_iter):
        structure = step["structure"]

        frames.append(Frame(
            index     = selected_idx[i],
            lattice   = np.array(structure.lattice.matrix),
            species   = [str(s) for s in structure.species],
            positions = np.array(structure.frac_coords),
            energy    = float(step[energy_key]),
            forces    = np.array(step["forces"]),
            stress    = _extract_stress(step),
            source    = str(path),
        ))

    if logger:
        logger.info(f"Frame extraction done in {time.time() - t_ext:.1f}s")

    return Dataset(
        frames  = frames,
        n_atoms = len(all_species),
        species = unique_species,
    )


if __name__ == "__main__":
    data = read_vasprun("vasprun.xml", frame_start=0, frame_end=-1)
    print(f"Total frames loaded : {len(data.frames)}")
    print(f"Number of atoms     : {data.n_atoms}")
    print(f"Species             : {data.species}")
