from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from ase.io import iread

from .dataset import Frame, Dataset


def _extract_energy(atoms) -> float:
    """Pull total energy in eV. extxyz commonly carries it in info['energy']
    or on a SinglePointCalculator; try both."""
    info = atoms.info
    for key in ("energy", "free_energy", "TotEnergy", "total_energy"):
        if key in info:
            return float(info[key])
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        raise ValueError(
            f"extxyz frame is missing total energy "
            f"(no 'energy' info key and no attached calculator)."
        ) from exc


def _extract_forces(atoms) -> np.ndarray:
    """Pull per-atom forces in eV/Å. Standard extxyz key is 'forces'."""
    arrays = atoms.arrays
    for key in ("forces", "force", "F"):
        if key in arrays:
            return np.array(arrays[key], dtype=float)
    try:
        return np.array(atoms.get_forces(), dtype=float)
    except Exception as exc:
        raise ValueError(
            f"extxyz frame is missing per-atom forces "
            f"(no 'forces' array and no attached calculator)."
        ) from exc


def _extract_stress(atoms) -> np.ndarray | None:
    """Optional 3x3 stress in the Frame convention (eV/Å³, ASE sign
    σ = (1/V) ∂E/∂ε). An extxyz 'stress' info key and ASE's calculator stress
    already follow that convention. A 'virial' key is the extensive tensor
    W = −V·σ (the convention MLFF trainers use), so it is converted via
    σ = −W/V."""
    info = atoms.info
    raw = None
    is_virial = False
    if "stress" in info:
        raw = np.array(info["stress"], dtype=float)
    elif "virial" in info:
        raw = np.array(info["virial"], dtype=float)
        is_virial = True
    if raw is None:
        try:
            raw = np.array(atoms.get_stress(voigt=False), dtype=float)
        except Exception:
            return None

    if raw.shape == (6,):
        xx, yy, zz, yz, xz, xy = raw
        raw = np.array([
            [xx, xy, xz],
            [xy, yy, yz],
            [xz, yz, zz],
        ])
    elif raw.shape == (9,):
        raw = raw.reshape(3, 3)
    elif raw.shape != (3, 3):
        return None

    if is_virial:
        raw = -raw / atoms.get_volume()
    return raw


def read_extxyz(
    filepath:    str,
    frame_start: int = 0,
    frame_end:   int = -1,
    stride:      int = 1,
    logger:      logging.Logger | None = None,
) -> Dataset:
    """
    Read an extxyz trajectory and return a Dataset.

    Required per-frame quantities: total energy (eV), per-atom forces (eV/Å),
    lattice vectors. Stress is optional. Positions are stored as fractional
    to match the convention used by vasp_reader / potentials.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"extxyz file not found: '{filepath}'.")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}.")

    if logger:
        logger.info(f"Parsing {path} with ASE ...")
    t0 = time.time()

    frames: list[Frame] = []
    ref_species: list[str] | None = None
    ref_unique:  list[str] | None = None

    last_end = float("inf") if frame_end == -1 else frame_end
    kept = 0

    for i, atoms in enumerate(iread(str(path), index=":")):
        if i < frame_start or i > last_end:
            continue
        if ((i - frame_start) % stride) != 0:
            continue

        cell = np.array(atoms.cell.array, dtype=float)  # (3, 3) Å
        if np.linalg.det(cell) == 0:
            raise ValueError(
                f"extxyz frame {i} has no valid cell. "
                f"Make sure the file was written with `pbc=True` and a Lattice."
            )

        species = [str(s) for s in atoms.get_chemical_symbols()]
        unique  = list(dict.fromkeys(species))

        if ref_species is None:
            ref_species = species
            ref_unique  = unique
        elif species != ref_species:
            raise ValueError(
                f"extxyz frame {i} has species order different from the first "
                f"frame. All frames in one file must share atom order."
            )

        cart_positions = np.array(atoms.get_positions(), dtype=float)   # (N, 3) Å
        frac_positions = cart_positions @ np.linalg.inv(cell)

        frames.append(Frame(
            index     = i,
            lattice   = cell,
            species   = species,
            positions = frac_positions,
            energy    = _extract_energy(atoms),
            forces    = _extract_forces(atoms),
            stress    = _extract_stress(atoms),
            source    = str(path),
        ))
        kept += 1

    if not frames:
        raise ValueError(
            f"No frames selected from '{filepath}' "
            f"(start={frame_start}, end={frame_end}, stride={stride})."
        )

    if logger:
        logger.info(
            f"Extracted {kept} frames from {path} in {time.time() - t0:.1f}s"
        )

    return Dataset(
        frames  = frames,
        n_atoms = len(ref_species),
        species = ref_unique,
    )


if __name__ == "__main__":
    import sys
    data = read_extxyz(sys.argv[1] if len(sys.argv) > 1 else "data.extxyz")
    print(f"Total frames loaded : {len(data.frames)}")
    print(f"Number of atoms     : {data.n_atoms}")
    print(f"Species             : {data.species}")
