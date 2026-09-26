"""
Physical invariance checks: shift all atoms by the same vector and verify

1. energy is unchanged (translation invariance), and
2. sum of forces is zero (Newton's third law).

Both follow from any short-range pairwise (and many-body-symmetric) potential
in a periodic cell, and are independent of the analytical force expression —
i.e. they would catch even subtle bugs in a chain-rule derivation.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset      import load_dataset, DatasetEntry
from src.potentials       import (
    Coulomb, Repulsive, BV, BVV, Angle,
    clear_neighbor_cache,
)
from src.extensions.ewald import Ewald, clear_kvec_cache


CUTOFF = 6.0


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/vasprun.xml",
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    return data.frames[0]


def _shift(positions: np.ndarray, lattice: np.ndarray, delta_cart: np.ndarray) -> np.ndarray:
    """Add the cartesian vector delta_cart to every atom, expressed in
    fractional coords. The cell origin moves with the atoms, so positions
    remain inside [0, 1)^3 after the standard PBC wrap."""
    delta_frac = delta_cart @ np.linalg.inv(lattice)
    return positions + delta_frac


def _check_invariance(pot, frame, delta_cart, e_atol=1e-9, f_atol=1e-9):
    lattice, species, positions = frame.lattice, frame.species, frame.positions

    e0 = pot.energy(lattice, species, positions)
    f0 = pot.forces(lattice, species, positions)

    shifted = _shift(positions, lattice, delta_cart)
    e1 = pot.energy(lattice, species, shifted)
    f1 = pot.forces(lattice, species, shifted)

    # Translation invariance of energy
    assert e1 == pytest.approx(e0, abs=e_atol), \
        f"energy changed under cartesian shift {delta_cart}: {e0} → {e1}"

    # Sum of forces is zero (both original and shifted)
    assert float(np.linalg.norm(f0.sum(axis=0))) == pytest.approx(0.0, abs=f_atol), \
        f"original force sum is non-zero: {f0.sum(axis=0)}"
    assert float(np.linalg.norm(f1.sum(axis=0))) == pytest.approx(0.0, abs=f_atol), \
        f"shifted force sum is non-zero: {f1.sum(axis=0)}"


_DELTAS = [
    np.array([0.1, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.3]),
    np.array([0.05, -0.2, 0.15]),
]


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_coulomb(frame, delta):
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    _check_invariance(coul, frame, delta, e_atol=1e-9, f_atol=1e-10)


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_repulsive(frame, delta):
    rep = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    _check_invariance(rep, frame, delta, e_atol=1e-12, f_atol=1e-12)


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_bv(frame, delta):
    bv_sp = {"Pb": {"V0": 2.0, "S": 0.5},
             "Ti": {"V0": 4.0, "S": 0.5},
             "O":  {"V0": 2.0, "S": 0.5}}
    bv_pp = {"O-Pb": {"r0": 2.06, "C": 6.0},
             "O-Ti": {"r0": 1.81, "C": 5.2}}
    bv = BV(species_params=bv_sp, pair_params=bv_pp, cutoff=CUTOFF)
    _check_invariance(bv, frame, delta, e_atol=1e-9, f_atol=1e-10)


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_bvv(frame, delta):
    bvv_sp = {"Pb": {"W0": 0.5, "D": 0.1},
              "Ti": {"W0": 0.3, "D": 0.1},
              "O":  {"W0": 0.0, "D": 0.0}}
    bv_pp  = {"O-Pb": {"r0": 2.06, "C": 6.0},
              "O-Ti": {"r0": 1.81, "C": 5.2}}
    bvv = BVV(species_params=bvv_sp, pair_params=bv_pp, cutoff=CUTOFF)
    _check_invariance(bvv, frame, delta, e_atol=1e-12, f_atol=1e-12)


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_ewald(frame, delta):
    ewald = Ewald(
        charges = {"Pb": 1.4, "Ti": 1.0, "O": -0.8},
        alpha   = 0.3, kmax = 5, cutoff = CUTOFF, epsilon = 1.0,
    )
    # Ewald energies are ~hundreds of eV → looser absolute tolerance.
    _check_invariance(ewald, frame, delta, e_atol=1e-6, f_atol=1e-9)


@pytest.mark.parametrize("delta", _DELTAS)
def test_invariance_angle(frame, delta):
    ang = Angle(k=0.001, cutoff=3.5)
    # Angle energies depend only on bond directions, so they are exactly
    # invariant under translation; force sum is zero by construction.
    _check_invariance(ang, frame, delta, e_atol=1e-10, f_atol=1e-10)
