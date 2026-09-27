"""
Correctness of the periodic neighbour list (``Potential._pbc_distances`` /
``_neighbors``).

These tests exist because the previous minimum-image implementation silently
dropped interactions whenever ``cutoff > L/2`` (true for the PbTiO3 example:
cell ~7.76 Å, cutoff 6.0 Å), and the finite-difference force tests could not
catch it — they compare the analytic force to the FD of the *same* (broken)
energy function, so they only verify ``f = -dE/dx`` self-consistency, not that
the neighbour list contains the right bonds.

The check here is independent: we compare the neighbour list against an explicit
brute-force enumeration of every periodic image within the cutoff. A regression
to minimum image (one image per pair) fails loudly, especially for the
``cutoff > L/2`` cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.dataset import load_dataset, DatasetEntry
from bvff.core.potentials  import Coulomb, clear_neighbor_cache


CUTOFF = 6.0


def _brute_force_pairs(cart, lattice, cutoff, amax=3):
    """All (i, j) directed image interactions with 0 < |r| <= cutoff, as a
    sorted multiset of rounded distances. Includes self-images (i == i, n != 0).
    """
    n = len(cart)
    L = np.asarray(lattice, dtype=float)
    dists = []
    for i in range(n):
        for j in range(n):
            for n1 in range(-amax, amax + 1):
                for n2 in range(-amax, amax + 1):
                    for n3 in range(-amax, amax + 1):
                        if i == j and n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        shift = n1 * L[0] + n2 * L[1] + n3 * L[2]
                        d = np.linalg.norm(cart[j] + shift - cart[i])
                        if 0.0 < d <= cutoff:
                            dists.append(d)
    return np.sort(np.round(dists, 6))


def _neighbor_dists(lattice, frac_positions, cutoff):
    """Sorted multiset of distances from the production neighbour list."""
    clear_neighbor_cache()
    pot = Coulomb(charges={"Pb": 0.0, "Ti": 0.0, "O": 0.0}, cutoff=cutoff)
    _, _, r_vecs = pot._neighbors(np.asarray(lattice, float), np.asarray(frac_positions, float), cutoff)
    return np.sort(np.round(np.linalg.norm(r_vecs, axis=1), 6))


@pytest.fixture(scope="module")
def frame(pto_300k):
    data = load_dataset(entries=[DatasetEntry(
        path=str(pto_300k),
        frame_start=100, frame_end=101, stride=1,
    )])
    return data.frames[0]


def test_neighbor_list_matches_brute_force_real_cell(frame):
    """The PbTiO3 cell has cutoff (6.0) > L/2 (~3.88), so minimum image is
    invalid. The neighbour list must reproduce the full brute-force multiset."""
    L = np.asarray(frame.lattice, dtype=float)
    assert CUTOFF > np.linalg.norm(L, axis=1).min() / 2, "test precondition: cutoff > L/2"

    cart = np.asarray(frame.positions) @ L
    brute = _brute_force_pairs(cart, L, CUTOFF)
    got   = _neighbor_dists(L, frame.positions, CUTOFF)

    assert got.size == brute.size, (
        f"neighbour count mismatch: list={got.size}, brute-force={brute.size} "
        f"(a minimum-image regression would give ~{brute.size // 2})"
    )
    np.testing.assert_allclose(got, brute, atol=1e-5)


def test_neighbor_list_counts_all_images_not_just_nearest(frame):
    """Guard the specific failure mode: with cutoff > L/2 many pairs have a
    second image inside the cutoff. The list must contain *more* entries than a
    one-image-per-pair (minimum-image) list would."""
    L = np.asarray(frame.lattice, dtype=float)
    cart = np.asarray(frame.positions) @ L

    n = len(cart)
    inv = np.linalg.inv(L)
    diff = cart[None, :, :] - cart[:, None, :]
    df = diff @ inv
    df -= np.round(df)
    min_image_dist = np.linalg.norm(df @ L, axis=2)
    np.fill_diagonal(min_image_dist, np.inf)
    min_image_count = int(np.sum((min_image_dist > 0) & (min_image_dist <= CUTOFF)))

    got = _neighbor_dists(L, frame.positions, CUTOFF)
    assert got.size > min_image_count, (
        f"neighbour list ({got.size}) is no larger than the minimum-image list "
        f"({min_image_count}) — the all-images fix appears to have regressed"
    )


@pytest.mark.parametrize("cell", [
    np.diag([4.0, 4.0, 4.0]),                       # cubic, cutoff = 1.5 L/2
    np.array([[5.0, 0, 0], [0.5, 4.0, 0], [0, 0.7, 6.0]]),  # triclinic, anisotropic
])
def test_neighbor_list_matches_brute_force_small_cells(cell):
    """Synthetic small cells where cutoff strongly exceeds L/2 — the regime
    minimum image gets wrong. A few atoms at arbitrary fractional positions."""
    rng = np.random.default_rng(0)
    frac = rng.random((5, 3))
    cart = frac @ cell
    brute = _brute_force_pairs(cart, cell, CUTOFF, amax=4)
    got   = _neighbor_dists(cell, frac, CUTOFF)
    assert got.size == brute.size, f"count mismatch: {got.size} vs {brute.size}"
    np.testing.assert_allclose(got, brute, atol=1e-5)
