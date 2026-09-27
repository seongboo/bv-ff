"""
Snapshot tests for each Potential term.

Pins energy + force summary statistics (norm, max-abs, sum) on a single
deterministic PbTiO3 frame (index 100 of examples/PbTiO3/pbtio3_222_300K.parquet) with
fixed inline parameters. The snapshot values were regenerated after the ASE
full-periodic-image neighbour-list fix (which replaced the minimum-image list
and now counts every in-cutoff image — for this cell ~2× the bonds), and are
cross-checked against the analytic-vs-finite-difference suites that prove
force = -dE/dx on the same neighbour list.

Parameters are inline (not read from parameters.toml) so that user-side
parameter changes do not break these tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.dataset import load_dataset, DatasetEntry
from bvff.core.potentials  import (
    Coulomb, Repulsive, BV, BVV, Angle,
    clear_neighbor_cache,
)
from bvff.core.extensions.ewald import Ewald, clear_kvec_cache


CUTOFF = 6.0
ATOL   = 1e-10  # absolute tolerance on summary statistics


# ──────────────────────────────────────────────
# Frame fixture
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def frame(pto_300k):
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = str(pto_300k),
        frame_start = 100,
        frame_end   = 101,
        stride      = 1,
    )])
    fr = data.frames[0]
    assert fr.index == 100
    assert len(fr.species) == 40
    return fr


def _check_array(arr: np.ndarray, expected_norm, expected_max, expected_sum):
    a = np.asarray(arr)
    assert np.isfinite(a).all(), "non-finite value in force array"
    assert np.linalg.norm(a)    == pytest.approx(expected_norm, abs=ATOL)
    assert float(np.max(np.abs(a))) == pytest.approx(expected_max, abs=ATOL)
    assert float(a.sum())       == pytest.approx(expected_sum,  abs=ATOL)


# ──────────────────────────────────────────────
# Coulomb (direct sum)
# ──────────────────────────────────────────────

def test_coulomb(frame):
    # Snapshot regenerated after the direct Coulomb term gained the missing
    # KE = 14.3996 eV·Å/e² conversion (energies now in eV, consistent with
    # Ewald and every other term) — values are exactly the old ones × 14.3996.
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    e = coul.energy(frame.lattice, frame.species, frame.positions)
    f = coul.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(-2.273678947543255e+02, abs=ATOL)
    _check_array(f, 8.698761019911815e+00, 2.015910160421671e+00, 5.107025913275720e-15)


# ──────────────────────────────────────────────
# Repulsive
# ──────────────────────────────────────────────

def test_repulsive(frame):
    rep = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    e = rep.energy(frame.lattice, frame.species, frame.positions)
    f = rep.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(2.611499101206271e-02, abs=ATOL)
    # forces() returns f = -∂E/∂x, scattered to atom i over the full directed
    # neighbour list (the (j,i) entry supplies j's force).
    _check_array(f, 3.453460483647281e-02, 1.059275605766025e-02, 1.691355389077387e-17)


# ──────────────────────────────────────────────
# Bond Valence
# ──────────────────────────────────────────────

_BV_SP = {"Pb": {"V0": 2.0, "S": 0.5},
          "Ti": {"V0": 4.0, "S": 0.5},
          "O":  {"V0": 2.0, "S": 0.5}}
_BV_PP = {"O-Pb": {"r0": 2.06, "C": 6.0},
          "O-Ti": {"r0": 1.81, "C": 5.2}}

def test_bv_valence(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    V  = bv.get_valence(frame.lattice, frame.species, frame.positions)
    _check_array(V, 1.739068306660597e+01, 4.364975897032004e+00, 1.056993634406171e+02)

def test_bv_energy_forces(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    e = bv.energy(frame.lattice, frame.species, frame.positions)
    f = bv.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(1.634839648715407e+00, abs=ATOL)
    # BV.forces includes both the V_i- and V_j-derivative (Newton 3rd-law)
    # contributions per pair; net force sum is ~0 (machine epsilon).
    _check_array(f, 5.757491291517670e+00, 1.788243343844266e+00, 4.440892098500626e-15)


# ──────────────────────────────────────────────
# Bond Valence Vector
# ──────────────────────────────────────────────

_BVV_SP = {"Pb": {"W0": 0.5, "D": 0.1},
           "Ti": {"W0": 0.3, "D": 0.1},
           "O":  {"W0": 0.0, "D": 0.0}}

def test_bvv_W(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    W   = bvv.get_bvv(frame.lattice, frame.species, frame.positions)
    _check_array(W, 3.357091983067578e+00, 8.968580766288343e-01, 1.665334536937735e-16)

def test_bvv_energy_forces(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    e   = bvv.energy(frame.lattice, frame.species, frame.positions)
    f   = bvv.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(4.227788433578131e-02, abs=ATOL)
    # BVV.forces uses the full 3×3 ∂(V r̂)/∂r_a tensor (radial + lateral +
    # Newton 3rd-law components); net force sum is ~0 (machine epsilon).
    _check_array(f, 5.319156668978221e-01, 1.841877790135539e-01, -1.804112415015879e-16)


# ──────────────────────────────────────────────
# Angle (O-centered, smaller cutoff so only nearest O-O-O triples)
# ──────────────────────────────────────────────

def test_angle(frame):
    # Snapshot regenerated after the triplet species filter fix: both LEGS are
    # now required to be angle_species too (previously only the center was
    # filtered, so Pb/Ti neighbors formed spurious O-centered triplets — e.g.
    # 90° Ti-O-Ti bridges penalized toward 180°). The old pinned value
    # (1.860e+04) included those unphysical triplets.
    ang = Angle(k=0.001, cutoff=3.5)
    e = ang.energy(frame.lattice, frame.species, frame.positions)
    f = ang.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(5.317361247392817e+03, abs=ATOL)
    # forces() returns f = -∂E/∂x.
    _check_array(f, 7.128689366822859e+01, 2.412008077844144e+01, -7.105427357601002e-15)


# ──────────────────────────────────────────────
# Caching invariance — same call twice must give bit-identical results
# ──────────────────────────────────────────────

def test_cache_invariance(frame):
    """Caching must not change values across repeated calls or after invalidation."""
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    bv   = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)

    e1 = coul.energy(frame.lattice, frame.species, frame.positions)
    f1 = bv.forces(frame.lattice, frame.species, frame.positions)
    e2 = coul.energy(frame.lattice, frame.species, frame.positions)
    f2 = bv.forces(frame.lattice, frame.species, frame.positions)
    clear_neighbor_cache()
    e3 = coul.energy(frame.lattice, frame.species, frame.positions)
    f3 = bv.forces(frame.lattice, frame.species, frame.positions)

    assert e1 == e2 == e3
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_equal(f1, f3)
