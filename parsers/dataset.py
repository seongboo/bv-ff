from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    """Single ionic step / configuration. Positions are fractional."""
    index:     int
    lattice:   np.ndarray          # (3, 3) lattice vectors in Angstrom
    species:   list[str]           # element symbols, length = N_atoms
    positions: np.ndarray          # (N, 3) fractional coordinates
    energy:    float               # total energy in eV
    forces:    np.ndarray          # (N, 3) forces in eV/Angstrom
    stress:    np.ndarray | None   # (3, 3) stress tensor, format-dependent units
    source:    str = ""            # originating file path (one trajectory / temperature per file)


@dataclass
class Dataset:
    """Collection of frames. Species list preserves order of first appearance."""
    frames:  list[Frame]
    n_atoms: int
    species: list[str]


@dataclass
class DatasetEntry:
    """One trajectory file with its own frame-selection window."""
    path:        str
    frame_start: int = 0
    frame_end:   int = -1   # -1 = all
    stride:      int = 1


# ──────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────

_VASP_EXTS    = {".xml"}
_EXTXYZ_EXTS  = {".xyz", ".extxyz"}


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _VASP_EXTS or path.name.lower().startswith("vasprun"):
        return "vasprun"
    if suffix in _EXTXYZ_EXTS:
        return "extxyz"
    raise ValueError(
        f"Cannot detect dataset format from '{path}'. "
        f"Use a '.xml' (vasprun) or '.xyz'/'.extxyz' file."
    )


# ──────────────────────────────────────────────
# Dispatcher
# ──────────────────────────────────────────────

def load_dataset(
    entries: list[DatasetEntry],
    logger:  logging.Logger | None = None,
) -> Dataset:
    """
    Load one or more trajectory files into a single Dataset.

    Each entry carries its own (path, frame_start, frame_end, stride). Frames
    are concatenated in the order given. Species and atom count must match
    across files.
    """
    if not entries:
        raise ValueError("dataset is empty.")

    # Local imports avoid a hard dependency loop between dataset.py and the
    # format-specific readers.
    from .vasp_reader   import read_vasprun
    from .extxyz_reader import read_extxyz

    all_frames: list[Frame] = []
    ref_species: list[str] | None = None
    ref_n_atoms: int | None = None

    for entry in entries:
        path = Path(entry.path)
        if not path.exists():
            raise FileNotFoundError(f"dataset file not found: '{entry.path}'.")
        if entry.stride < 1:
            raise ValueError(f"stride must be >= 1, got {entry.stride} (file '{entry.path}').")

        fmt = _detect_format(path)
        if logger:
            logger.info(
                f"Loading {path} (format={fmt}, start={entry.frame_start}, "
                f"end={entry.frame_end}, stride={entry.stride}) ..."
            )

        reader = read_vasprun if fmt == "vasprun" else read_extxyz
        sub = reader(
            str(path),
            frame_start = entry.frame_start,
            frame_end   = entry.frame_end,
            stride      = entry.stride,
            logger      = logger,
        )

        if ref_species is None:
            ref_species = sub.species
            ref_n_atoms = sub.n_atoms
        else:
            if sub.species != ref_species:
                raise ValueError(
                    f"Species mismatch between files: {ref_species} vs {sub.species} "
                    f"(file '{entry.path}')."
                )
            if sub.n_atoms != ref_n_atoms:
                raise ValueError(
                    f"Atom count mismatch between files: {ref_n_atoms} vs {sub.n_atoms} "
                    f"(file '{entry.path}')."
                )

        all_frames.extend(sub.frames)

    if logger:
        logger.info(f"Loaded {len(all_frames)} total frames from {len(entries)} file(s).")

    return Dataset(
        frames  = all_frames,
        n_atoms = ref_n_atoms,
        species = ref_species,
    )
