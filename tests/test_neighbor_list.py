"""
Neighbor list vs an independent reference (ASE ``neighbor_list``).

Potential._pbc_distances must return every (i, j, n) with
|r_j + n·L - r_i| <= r_c over all periodic images, including self-images
(i == j, n != 0), as a full ordered list. ASE builds the same list by a
different algorithm (spatial binning), so agreement of the complete
(i, j, D) multiset is a strong check. Cases cover r_c > half the cell
width (where minimum image fails), r_c > the full cell width (several
image shells), a skewed triclinic cell, and a one-atom cell (self-images
only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from parsers.dataset  import load_dataset, DatasetEntry
from src.potentials   import Potential, clear_neighbor_cache


_ROOT = Path(__file__).resolve().parent.parent


class _Probe(Potential):
    """Minimal concrete Potential to reach the neighbor-list methods."""
    def energy(self, lattice, species, positions):  return 0.0
    def forces(self, lattice, species, positions):  return np.zeros((len(species), 3))


def _canonical(i, j, D, decimals=8):
    """Sort (i, j, D) rows lexicographically so two lists can be compared
    regardless of emission order."""
    rows = np.column_stack([i, j, np.round(D, decimals)])
    return rows[np.lexsort(rows.T[::-1])]


def _compare(lattice, frac, cutoff):
    clear_neighbor_cache()
    lattice = np.asarray(lattice, dtype=float)
    frac    = np.asarray(frac, dtype=float)
    i, j, D = _Probe()._neighbors(lattice, frac, cutoff)

    atoms = Atoms(numbers=[1] * len(frac), scaled_positions=frac,
                  cell=lattice, pbc=True)
    ia, ja, Da = neighbor_list("ijD", atoms, cutoff)

    assert len(i) == len(ia), f"pair count {len(i)} vs ASE {len(ia)}"
    np.testing.assert_allclose(_canonical(i, j, D), _canonical(ia, ja, Da),
                               atol=1e-7)
    return len(i)


def test_pbtio3_frame_cutoff_above_half_width():
    fr = load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=101, stride=1,
    )]).frames[0]
    # half perpendicular widths are 3.84 / 3.84 / 4.75 Å
    _compare(fr.lattice, fr.positions, 6.0)


def test_triclinic_cutoff_above_full_width():
    rng = np.random.default_rng(0)
    lattice = np.array([[4.0, 0.0, 0.0],
                        [1.7, 3.6, 0.0],
                        [-1.1, 0.9, 3.3]])
    frac = rng.random((7, 3))
    # r_c exceeds every perpendicular width → ≥ 2 image shells per axis
    _compare(lattice, frac, 7.5)


def test_single_atom_self_images():
    # one atom in a simple-cubic cell: neighbors are its own images only
    a = 3.0
    n = _compare(np.eye(3) * a, [[0.1, 0.2, 0.3]], 3.1)
    assert n == 6          # ±a along x, y, z
    n = _compare(np.eye(3) * a, [[0.1, 0.2, 0.3]], 4.3)
    assert n == 18         # + 12 face diagonals at a√2 = 4.24 Å


def test_wrapped_and_unwrapped_positions_agree():
    fr = load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=101, stride=1,
    )]).frames[0]
    shifted = fr.positions + np.array([[2, -1, 3]] * len(fr.positions))
    p = _Probe()
    clear_neighbor_cache()
    a = _canonical(*p._neighbors(fr.lattice, fr.positions, 6.0))
    b = _canonical(*p._neighbors(fr.lattice, shifted, 6.0))
    np.testing.assert_allclose(a, b, atol=1e-9)
