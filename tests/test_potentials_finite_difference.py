"""
Finite-difference check: analytical forces equal -∂E/∂x to ~FD precision.

For each potential, pick one atom per species (Pb, Ti, O) and verify

    forces(lattice, species, positions)[atom, axis]
        ≈ -(E(pos + ε e_axis) - E(pos - ε e_axis)) / (2ε)

This is the test that would have caught the Repulsive/Ewald-real double-count
bug (forces were 2× the gradient).
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
EPS    = 1e-5     # central-difference step in cartesian Å


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/vasprun.xml",
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    fr = data.frames[0]
    # 2×2×2 supercell: indices 0–7 = Pb, 8–15 = Ti, 16–39 = O
    assert fr.species[0]  == "Pb"
    assert fr.species[8]  == "Ti"
    assert fr.species[16] == "O"
    return fr


# ──────────────────────────────────────────────
# Finite-difference helper
# ──────────────────────────────────────────────

def _fd_force(pot, lattice, species, positions, atom, axis, eps=EPS):
    """Central-difference estimate of -dE/dx_axis at atom in cartesian."""
    inv_lat = np.linalg.inv(lattice)
    # Perturbing cart by ε e_axis ⇔ adding ε * inv(L)[axis, :] to frac coords.
    delta_frac = eps * inv_lat[axis, :]
    pos_plus              = positions.copy()
    pos_plus[atom]       += delta_frac
    pos_minus             = positions.copy()
    pos_minus[atom]      -= delta_frac
    # Distinct array identities → cache miss, fresh compute.
    e_plus  = pot.energy(lattice, species, pos_plus)
    e_minus = pot.energy(lattice, species, pos_minus)
    return -(e_plus - e_minus) / (2.0 * eps)


def _check_fd(pot, frame, atoms, atol):
    f_analytic = pot.forces(frame.lattice, frame.species, frame.positions)
    for a in atoms:
        for k in range(3):
            f_fd = _fd_force(pot, frame.lattice, frame.species, frame.positions, a, k)
            assert f_fd == pytest.approx(f_analytic[a, k], abs=atol), (
                f"atom={a} ({frame.species[a]}), axis={k}: "
                f"analytical={f_analytic[a, k]:.6e}, FD={f_fd:.6e}"
            )


# Sample atoms — one per species.
_ATOMS = [0, 8, 16]


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

def test_fd_coulomb(frame):
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    _check_fd(coul, frame, _ATOMS, atol=1e-5)


def test_fd_repulsive(frame):
    rep = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    _check_fd(rep, frame, _ATOMS, atol=1e-6)


def test_fd_bv(frame):
    bv_sp = {"Pb": {"V0": 2.0, "S": 0.5},
             "Ti": {"V0": 4.0, "S": 0.5},
             "O":  {"V0": 2.0, "S": 0.5}}
    bv_pp = {"O-Pb": {"r0": 2.06, "C": 6.0},
             "O-Ti": {"r0": 1.81, "C": 5.2}}
    bv = BV(species_params=bv_sp, pair_params=bv_pp, cutoff=CUTOFF)
    _check_fd(bv, frame, _ATOMS, atol=1e-5)


def test_fd_bvv(frame):
    bvv_sp = {"Pb": {"W0": 0.5, "D": 0.1},
              "Ti": {"W0": 0.3, "D": 0.1},
              "O":  {"W0": 0.0, "D": 0.0}}
    bv_pp  = {"O-Pb": {"r0": 2.06, "C": 6.0},
              "O-Ti": {"r0": 1.81, "C": 5.2}}
    bvv = BVV(species_params=bvv_sp, pair_params=bv_pp, cutoff=CUTOFF)
    _check_fd(bvv, frame, _ATOMS, atol=1e-5)


def test_fd_ewald(frame):
    ewald = Ewald(
        charges = {"Pb": 1.4, "Ti": 1.0, "O": -0.8},
        cutoff  = CUTOFF,   # tin-foil, α and k_c from default accuracy
    )
    # Ewald energy is on the ~100 eV scale → larger absolute FD error.
    _check_fd(ewald, frame, _ATOMS, atol=1e-3)


def test_fd_angle(frame):
    """Angle gradients are noisier near θ→180°; relaxed tolerance."""
    ang = Angle(k=0.001, cutoff=3.5)
    # Only test O atoms — Angle is O-centered; on Pb/Ti the analytical force
    # may be zero or near it, but FD still works. Just keep O sample.
    _check_fd(ang, frame, [16, 24, 32], atol=1e-2)
