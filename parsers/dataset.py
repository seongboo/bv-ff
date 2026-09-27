from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from math import gcd
from pathlib import Path

import numpy as np


# 1 eV/Å³ = 1602.1766208 kBar. Readers use this to normalize VASP-convention
# kBar stress into the single Frame.stress convention (eV/Å³, ASE sign).
KBAR_PER_EV_PER_ANG3 = 1602.1766208


def _composition_ratio(species: list[str]) -> tuple[tuple[str, int], ...]:
    """Reduced composition ratio, e.g. 40-atom PbTiO3 (8,8,24) → (('O',3),('Pb',1),('Ti',1)).
    Used so frames of the same chemistry but different supercell size (e.g. a
    5-atom DFT anchor and a 40-atom AIMD cell) are accepted together — the
    per-atom energy offset only requires a common composition *ratio*."""
    c = Counter(species)
    g = reduce(gcd, c.values()) if c else 1
    return tuple(sorted((el, n // g) for el, n in c.items()))


@dataclass
class Frame:
    """Single ionic step / configuration. Positions are fractional."""
    index:     int
    lattice:   np.ndarray          # (3, 3) lattice vectors in Angstrom
    species:   list[str]           # element symbols, length = N_atoms
    positions: np.ndarray          # (N, 3) fractional coordinates
    energy:    float               # total energy in eV
    forces:    np.ndarray          # (N, 3) forces in eV/Angstrom
    # (3, 3) stress tensor in eV/Å³, ASE sign convention σ = (1/V) ∂E/∂ε.
    # Every reader normalizes to this ONE convention (vasp/parquet convert
    # from VASP's kBar with the sign flip; extxyz is already ASE), so the
    # loss can compare against the BVFF virial without format-specific code.
    stress:    np.ndarray | None
    source:    str = ""            # originating file path (train/test stratification key)
    # Multi-source fitting (see fitting.compute_loss). ref_group is the
    # energy-offset unit: one constant offset is removed per ref_group, so frames
    # from different DFT references (e.g. AIMD Γ-only vs DFT 4×4×4) don't bias
    # each other, while energy differences *within* a group are fit. weight_* are
    # per-channel loss weights. split: True → ratio split, False → pinned to
    # train (sparse DFT anchor), "test" → pinned to test (hold out a whole
    # file/condition, e.g. leave-one-temperature-out).
    ref_group: str   = "default"
    weight_E:  float = 1.0
    weight_F:  float = 1.0
    weight_S:  float = 1.0
    split:     bool | str = True


@dataclass
class Dataset:
    """Collection of frames. Species list preserves order of first appearance."""
    frames:  list[Frame]
    n_atoms: int
    species: list[str]


@dataclass
class DatasetEntry:
    """One trajectory file with its own frame-selection window + fit metadata."""
    path:        str
    frame_start: int = 0
    frame_end:   int = -1   # -1 = all
    stride:      int = 1
    # Stamped onto every Frame this entry produces (see Frame). split: True →
    # ratio split, False → all to train, "test" → all to test.
    ref_group: str   = "default"
    weight_E:  float = 1.0
    weight_F:  float = 1.0
    weight_S:  float = 1.0
    split:     bool | str = True


# ──────────────────────────────────────────────
# Format detection
# ──────────────────────────────────────────────

_VASP_EXTS    = {".xml"}
_EXTXYZ_EXTS  = {".xyz", ".extxyz"}
_PARQUET_EXTS = {".parquet", ".pq"}


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _VASP_EXTS or path.name.lower().startswith("vasprun"):
        return "vasprun"
    if suffix in _EXTXYZ_EXTS:
        return "extxyz"
    if suffix in _PARQUET_EXTS:
        return "parquet"
    raise ValueError(
        f"Cannot detect dataset format from '{path}'. "
        f"Use a '.xml' (vasprun), '.xyz'/'.extxyz', or '.parquet' file."
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

    # Import each format reader lazily, only when that format is actually used.
    # This avoids a hard dependency loop between dataset.py and the readers, and
    # — since the readers pull in heavy optional deps (pymatgen for vasprun, ASE
    # for extxyz, pandas for parquet) — keeps a run that uses only one format
    # from requiring the others' dependencies to be installed.
    def _get_reader(fmt: str):
        if fmt == "vasprun":
            from .vasp_reader import read_vasprun
            return read_vasprun
        if fmt == "extxyz":
            from .extxyz_reader import read_extxyz
            return read_extxyz
        if fmt == "parquet":
            from .parquet_reader import read_parquet
            return read_parquet
        raise ValueError(f"No reader for format '{fmt}'.")

    all_frames: list[Frame] = []
    ref_species: list[str] | None = None
    ref_n_atoms: int | None = None
    ref_ratio:   tuple | None = None

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

        reader = _get_reader(fmt)
        sub = reader(
            str(path),
            frame_start = entry.frame_start,
            frame_end   = entry.frame_end,
            stride      = entry.stride,
            logger      = logger,
        )

        sub_ratio = _composition_ratio(sub.species)
        if ref_species is None:
            ref_species = sub.species
            ref_n_atoms = sub.n_atoms
            ref_ratio   = sub_ratio
        elif sub_ratio != ref_ratio:
            raise ValueError(
                f"Composition ratio mismatch between files: "
                f"{dict(Counter(ref_species))} vs {dict(Counter(sub.species))} "
                f"(file '{entry.path}'). The per-atom energy reference requires a "
                f"single composition ratio across all frames."
            )

        # Stamp the entry's fit metadata onto every frame it produced.
        for fr in sub.frames:
            fr.ref_group = entry.ref_group
            fr.weight_E  = entry.weight_E
            fr.weight_F  = entry.weight_F
            fr.weight_S  = entry.weight_S
            fr.split     = entry.split

        all_frames.extend(sub.frames)

    # Reject non-finite (NaN/Inf) data up front: a single bad frame silently
    # poisons the RMSE loss (→ NaN) and sends the optimizer wandering.
    for fr in all_frames:
        for name, arr in (("energy", fr.energy), ("forces", fr.forces),
                          ("lattice", fr.lattice), ("positions", fr.positions),
                          ("stress", fr.stress)):
            if arr is None:
                continue
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"Non-finite {name} in frame index={fr.index} "
                    f"(source='{fr.source}'). Check the dataset file."
                )

    # Fail fast on an empty load: otherwise `ref_n_atoms`/`ref_species` stay None and
    # Dataset(frames=[], n_atoms=None, ...) crashes obscurely far downstream (fit() on
    # train_frames[0], or None arithmetic) instead of here at the cause.
    if not all_frames or ref_species is None or ref_n_atoms is None:
        raise ValueError(
            "No frames were loaded from the dataset (all file/frame selections were "
            "empty). Check the dataset path(s) and any frame-window/stride settings."
        )

    if logger:
        logger.info(f"Loaded {len(all_frames)} total frames from {len(entries)} file(s).")

    return Dataset(
        frames  = all_frames,
        n_atoms = ref_n_atoms,
        species = ref_species,
    )
