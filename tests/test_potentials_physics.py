"""
Tests for the extended physics models:

  - Buckingham (Born-Mayer + dispersion) pair potential
  - exponential Brown-Altermatt bond-valence form (form="exp") for BV / BVV

Each model is checked for analytical force = -∂E/∂x (finite difference),
translation invariance (energy unchanged + zero net force under a rigid
shift), and — for the pairwise Buckingham — analytic virial = FD strain
derivative.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials  import Buckingham, BV, BVV, clear_neighbor_cache


CUTOFF = 6.0
EPS    = 1e-5
_ATOMS = [0, 8, 16]   # one Pb, one Ti, one O

_BUCK = {"O-O":  {"A": 1500.0, "rho": 0.30, "C": 12.0},
         "O-Pb": {"A": 2000.0, "rho": 0.35, "C": 18.0},
         "O-Ti": {"A":  877.0, "rho": 0.38, "C":  9.0}}

_BV_PP_EXP = {"O-Pb": {"r0": 2.06, "C": 6.0, "b": 0.37},
              "O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.40}}
_BV_SP  = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BVV_SP = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.0, "D": 0.0}}


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    data = load_dataset(entries=[DatasetEntry(
        path="examples/PbTiO3/vasprun.xml", frame_start=100, frame_end=101, stride=1,
    )])
    return data.frames[0]


def _fd_force(pot, lattice, species, positions, atom, axis, eps=EPS):
    inv_lat = np.linalg.inv(lattice)
    delta_frac = eps * inv_lat[axis, :]
    pos_plus  = positions.copy(); pos_plus[atom]  += delta_frac
    pos_minus = positions.copy(); pos_minus[atom] -= delta_frac
    e_plus  = pot.energy(lattice, species, pos_plus)
    e_minus = pot.energy(lattice, species, pos_minus)
    return -(e_plus - e_minus) / (2.0 * eps)


def _check_fd(pot, frame, atoms, atol):
    f = pot.forces(frame.lattice, frame.species, frame.positions)
    for a in atoms:
        for k in range(3):
            f_fd = _fd_force(pot, frame.lattice, frame.species, frame.positions, a, k)
            assert f_fd == pytest.approx(f[a, k], abs=atol), (
                f"atom={a} ({frame.species[a]}), axis={k}: analytic={f[a, k]:.6e}, FD={f_fd:.6e}"
            )


def _fd_stress(pot, lattice, species, positions, eps=1e-4):
    lat = np.asarray(lattice, dtype=float); V = abs(np.linalg.det(lat))
    s = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            F = np.eye(3); F[a, b] += eps
            ep = pot.energy(lat @ F.T, species, positions)
            F[a, b] -= 2 * eps
            em = pot.energy(lat @ F.T, species, positions)
            s[a, b] = (ep - em) / (2 * eps) / V
    return 0.5 * (s + s.T)


# ──────────────────────────────────────────────
# Buckingham
# ──────────────────────────────────────────────

def test_buckingham_fd(frame):
    buck = Buckingham(params=_BUCK, cutoff=CUTOFF)
    _check_fd(buck, frame, _ATOMS, atol=1e-4)


def test_buckingham_translation_invariance(frame):
    buck = Buckingham(params=_BUCK, cutoff=CUTOFF)
    e0 = buck.energy(frame.lattice, frame.species, frame.positions)
    f0 = buck.forces(frame.lattice, frame.species, frame.positions)
    shifted = frame.positions + np.array([0.07, -0.13, 0.21]) @ np.linalg.inv(frame.lattice)
    e1 = buck.energy(frame.lattice, frame.species, shifted)
    assert e1 == pytest.approx(e0, abs=1e-9)
    assert float(np.linalg.norm(f0.sum(axis=0))) == pytest.approx(0.0, abs=1e-9)


def test_buckingham_stress_analytic_vs_fd(frame):
    # Cutoff in a pair-distance gap; see STRESS_CUTOFF in test_potentials_stress.py.
    buck = Buckingham(params=_BUCK, cutoff=5.90)
    analytic = buck.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(buck, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-6)
    assert np.allclose(analytic, analytic.T)


# ──────────────────────────────────────────────
# Exponential bond-valence form
# ──────────────────────────────────────────────

def test_bv_exp_fd(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP_EXP, cutoff=CUTOFF, form="exp")
    _check_fd(bv, frame, _ATOMS, atol=1e-5)


def test_bvv_exp_fd(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP_EXP, cutoff=CUTOFF, form="exp")
    _check_fd(bvv, frame, _ATOMS, atol=1e-5)


def test_bv_exp_energy_and_forces_consistent(frame):
    """The one-pass energy_and_forces must match the separate energy/forces."""
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP_EXP, cutoff=CUTOFF, form="exp")
    e1 = bv.energy(frame.lattice, frame.species, frame.positions)
    f1 = bv.forces(frame.lattice, frame.species, frame.positions)
    e2, f2 = bv.energy_and_forces(frame.lattice, frame.species, frame.positions)
    assert e1 == pytest.approx(e2, abs=1e-12)
    np.testing.assert_allclose(f1, f2, atol=1e-12)


def test_bv_exp_translation_invariance(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP_EXP, cutoff=CUTOFF, form="exp")
    e0 = bv.energy(frame.lattice, frame.species, frame.positions)
    f0 = bv.forces(frame.lattice, frame.species, frame.positions)
    shifted = frame.positions + np.array([0.05, -0.2, 0.15]) @ np.linalg.inv(frame.lattice)
    e1 = bv.energy(frame.lattice, frame.species, shifted)
    assert e1 == pytest.approx(e0, abs=1e-9)
    assert float(np.linalg.norm(f0.sum(axis=0))) == pytest.approx(0.0, abs=1e-9)


def test_bv_invalid_form_raises():
    with pytest.raises(ValueError):
        BV(species_params=_BV_SP, pair_params=_BV_PP_EXP, cutoff=CUTOFF, form="nope")
