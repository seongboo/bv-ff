"""
Whole-potential (BVFF) correctness, beyond the per-term checks.

The per-term suites pin each Potential in isolation. These tests pin the
*combination*: that the aggregate energy/forces are the sum of the parts,
that the single-pass ``energy_and_forces`` matches the separate calls, and
that the combined force field is still the exact gradient of the combined
energy (finite difference). A sign or double-count bug that happened to
cancel within one term but not across the sum would show up here.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.dataset      import load_dataset, DatasetEntry
from bvff.core.potentials       import (
    Coulomb, Repulsive, BV, BVV, BVFF, clear_neighbor_cache,
)
from bvff.core.extensions.ewald import Ewald, clear_kvec_cache


CUTOFF = 6.0
EPS    = 1e-5

_CHARGES = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}
_B       = {"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}
_BV_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BV_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}
_BVV_SP  = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.0, "D": 0.0}}


@pytest.fixture(scope="module")
def frame(pto_300k):
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path=str(pto_300k), frame_start=100, frame_end=101, stride=1,
    )])
    return data.frames[0]


def _make_bvff():
    return BVFF([
        Coulomb(_CHARGES, CUTOFF),
        Repulsive(_B, CUTOFF),
        BV(_BV_SP, _BV_PP, CUTOFF),
        BVV(_BVV_SP, _BV_PP, CUTOFF),
    ])


def test_total_energy_is_sum_of_terms(frame):
    bvff = _make_bvff()
    total = bvff.energy(frame.lattice, frame.species, frame.positions)
    parts = sum(t.energy(frame.lattice, frame.species, frame.positions) for t in bvff.terms)
    assert total == pytest.approx(parts, abs=1e-12)


def test_total_forces_is_sum_of_terms(frame):
    bvff = _make_bvff()
    total = bvff.forces(frame.lattice, frame.species, frame.positions)
    parts = sum(t.forces(frame.lattice, frame.species, frame.positions) for t in bvff.terms)
    np.testing.assert_allclose(total, parts, atol=1e-12)


def test_total_energy_and_forces_consistent(frame):
    """One-pass energy_and_forces must equal the separate energy()/forces()."""
    bvff = _make_bvff()
    e1 = bvff.energy(frame.lattice, frame.species, frame.positions)
    f1 = bvff.forces(frame.lattice, frame.species, frame.positions)
    e2, f2 = bvff.energy_and_forces(frame.lattice, frame.species, frame.positions)
    assert e1 == pytest.approx(e2, abs=1e-12)
    np.testing.assert_allclose(f1, f2, atol=1e-12)


def test_total_forces_match_fd_all_atoms(frame):
    """Combined force field == -∂E_total/∂x for EVERY atom (not just a sample)."""
    bvff = _make_bvff()
    L, sp, X = frame.lattice, frame.species, frame.positions
    inv = np.linalg.inv(L)
    f_analytic = bvff.forces(L, sp, X)
    for a in range(len(sp)):
        for k in range(3):
            d = EPS * inv[k, :]
            xp = X.copy(); xp[a] += d
            xm = X.copy(); xm[a] -= d
            f_fd = -(bvff.energy(L, sp, xp) - bvff.energy(L, sp, xm)) / (2 * EPS)
            assert f_fd == pytest.approx(f_analytic[a, k], abs=1e-5), (
                f"atom={a} ({sp[a]}) axis={k}: analytic={f_analytic[a,k]:.3e} fd={f_fd:.3e}"
            )


def test_total_net_force_zero(frame):
    bvff = _make_bvff()
    f = bvff.forces(frame.lattice, frame.species, frame.positions)
    assert float(np.linalg.norm(f.sum(axis=0))) == pytest.approx(0.0, abs=1e-9)


def test_total_with_ewald_matches_fd(frame):
    """Same gradient check but with the Ewald term in the mix (looser tol:
    Ewald energies are on the ~100 eV scale)."""
    bvff = BVFF([Ewald(_CHARGES, alpha=0.3, kmax=5, cutoff=CUTOFF),
                 Repulsive(_B, CUTOFF), BV(_BV_SP, _BV_PP, CUTOFF)])
    L, sp, X = frame.lattice, frame.species, frame.positions
    inv = np.linalg.inv(L)
    fa = bvff.forces(L, sp, X)
    for a in (0, 8, 16, 24, 32):
        for k in range(3):
            d = EPS * inv[k, :]
            xp = X.copy(); xp[a] += d
            xm = X.copy(); xm[a] -= d
            f_fd = -(bvff.energy(L, sp, xp) - bvff.energy(L, sp, xm)) / (2 * EPS)
            assert f_fd == pytest.approx(fa[a, k], abs=1e-3)
