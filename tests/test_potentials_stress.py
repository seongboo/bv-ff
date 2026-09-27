"""
Virial stress tests.

The stress convention is σ_αβ = (1/V) ∂E/∂ε_αβ (eV/Å³). Two independent
checks per term:

1. Closed-form vs finite difference: EVERY built-in term (Coulomb, Repulsive,
   Buckingham, BV, BVV, Angle, Ewald) overrides the base-class FD strain
   derivative with an analytic virial. Each must agree with a hand-rolled FD
   computed here (independent of the implementation), pinning the sign and
   the 1/V factor — including with the cutoff taper active (the S'(r)
   chain-rule term enters the virial too).
2. Symmetry + translation invariance: the stress tensor is symmetric, and
   rigidly shifting every atom leaves it unchanged (it depends only on
   relative positions).
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset      import load_dataset, DatasetEntry
from src.potentials       import (
    Coulomb, Repulsive, Buckingham, BV, BVV, Angle, BVFF,
    clear_neighbor_cache,
)
from src.extensions.ewald import Ewald, clear_kvec_cache


CUTOFF = 6.0


# ──────────────────────────────────────────────
# Fixture
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/pbtio3_222_300K.parquet",
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    return data.frames[0]


def _fd_stress(pot, lattice, species, positions, eps=1e-4):
    """Independent central-difference strain derivative σ = (1/V) dE/dε.
    Deforms only the lattice (fractional positions are strain-invariant)."""
    lat = np.asarray(lattice, dtype=float)
    volume = abs(np.linalg.det(lat))
    sigma = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            F = np.eye(3)
            F[a, b] += eps
            e_plus = pot.energy(lat @ F.T, species, positions)
            F[a, b] -= 2.0 * eps
            e_minus = pot.energy(lat @ F.T, species, positions)
            sigma[a, b] = (e_plus - e_minus) / (2.0 * eps) / volume
    return 0.5 * (sigma + sigma.T)


def _shift(positions, lattice, delta_cart):
    return positions + delta_cart @ np.linalg.inv(lattice)


_BV_SP = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BV_PP = {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}
_BVV_SP = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.0, "D": 0.0}}


# ──────────────────────────────────────────────
# Analytic vs finite-difference (Coulomb, Repulsive)
# ──────────────────────────────────────────────

def test_stress_coulomb_analytic_vs_fd(frame):
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    analytic = coul.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(coul, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)
    assert np.allclose(analytic, analytic.T)


def test_stress_repulsive_analytic_vs_fd(frame):
    rep = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    analytic = rep.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(rep, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)
    assert np.allclose(analytic, analytic.T)


# ──────────────────────────────────────────────
# Analytic vs finite-difference (BV, BVV, Buckingham, Angle, Ewald)
# ──────────────────────────────────────────────

_BUCK = {"O-Ti": {"A": 700.0, "rho": 0.35, "C": 30.0},
         "O-Pb": {"A": 900.0, "rho": 0.38, "C": 20.0}}


def test_stress_buckingham_analytic_vs_fd(frame):
    buck = Buckingham(params=_BUCK, cutoff=CUTOFF)
    analytic = buck.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(buck, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)


def test_stress_bv_analytic_vs_fd(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    analytic = bv.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(bv, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)


def test_stress_bvv_analytic_vs_fd(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    analytic = bvv.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(bvv, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)


def test_stress_angle_analytic_vs_fd(frame):
    ang = Angle(k=1e-5, cutoff=3.5)
    analytic = ang.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(ang, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-6)
    assert np.abs(analytic).max() > 1e-6   # non-trivial comparison


@pytest.mark.parametrize("charges,epsilon", [
    ({"Pb": 1.4, "Ti": 1.0, "O": -0.8}, float("inf")),   # neutral, tinfoil
    ({"Pb": 1.4, "Ti": 1.0, "O": -0.8}, 2.0),            # neutral, dielectric surface
    ({"Pb": 1.4, "Ti": 1.0, "O": -0.7}, float("inf")),   # net charge → background
])
def test_stress_ewald_analytic_vs_fd(frame, charges, epsilon):
    """Covers all strain-dependent Ewald pieces: real, reciprocal, surface
    (finite epsilon), and neutralizing background (charged cell). kmax is
    fixed so the ±ε FD lattices can't flip the adaptive k-grid."""
    clear_kvec_cache()
    ew = Ewald(charges=charges, alpha=0.35, kmax=6, cutoff=CUTOFF, epsilon=epsilon)
    analytic = ew.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(ew, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-5)
    assert np.allclose(analytic, analytic.T)


@pytest.mark.parametrize("make", [
    lambda: Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF,
                    smooth_width=1.0),
    lambda: Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF,
                      smooth_width=1.0),
    lambda: Buckingham(params=_BUCK, cutoff=CUTOFF, smooth_width=1.0),
    lambda: BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
               smooth_width=1.0),
    lambda: BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
                smooth_width=1.0),
    lambda: Angle(k=1e-5, cutoff=3.5, smooth_width=0.5),
])
def test_stress_analytic_vs_fd_with_taper(frame, make):
    """The cutoff taper's S'(r) term must be included in the virial."""
    pot = make()
    analytic = pot.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(pot, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-6), type(pot).__name__


# ──────────────────────────────────────────────
# Symmetry + translation invariance (all terms)
# ──────────────────────────────────────────────

@pytest.mark.parametrize("delta", [np.array([0.1, 0.0, 0.0]), np.array([0.05, -0.2, 0.15])])
def test_stress_translation_invariance(frame, delta):
    terms = [
        Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF),
        Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF),
        BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF),
        BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF),
        Ewald(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, alpha=0.3, kmax=5, cutoff=CUTOFF),
    ]
    for pot in terms:
        s0 = pot.stress(frame.lattice, frame.species, frame.positions)
        shifted = _shift(frame.positions, frame.lattice, delta)
        s1 = pot.stress(frame.lattice, frame.species, shifted)
        assert np.allclose(s0, s0.T, atol=1e-8), f"{type(pot).__name__} stress not symmetric"
        assert np.allclose(s0, s1, atol=1e-6), f"{type(pot).__name__} stress changed under shift"


def test_bvff_total_stress_is_sum(frame):
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    rep  = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    bv   = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    bvff = BVFF([coul, rep, bv])

    total = bvff.stress(frame.lattice, frame.species, frame.positions)
    parts = sum(t.stress(frame.lattice, frame.species, frame.positions) for t in bvff.terms)
    assert np.allclose(total, parts)
    assert np.allclose(total, total.T)
