"""
Snapshot tests for each Potential term.

Pins energy + force summary statistics (norm, max-abs, sum) on a single
deterministic PbTiO3 frame (index 100 of examples/PbTiO3/vasprun.xml) with
fixed inline parameters. The snapshot values were captured after the
neighbor-cache + vectorisation refactor and cross-checked against the
pre-refactor Python-loop implementations to ~1e-13.

Parameters are inline (not read from parameters.toml) so that user-side
parameter changes do not break these tests.

Snapshot update (full periodic-image neighbor list): the neighbor list
previously used the minimum-image convention, which with CUTOFF = 6.0 Å
(> half the perpendicular cell widths, 3.84 / 3.84 / 4.75 Å) silently
dropped ~40 % of the pairs inside the cutoff (1442 → 2400 ordered pairs).
The Coulomb, Repulsive, BV and BVV snapshots were re-captured with the
all-images list, which is validated independently against ASE in
tests/test_neighbor_list.py and by the supercell-invariance tests in
tests/test_potentials_physical_correctness.py. Angle uses cutoff 3.5 Å
(< half width), so its snapshot is unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials  import (
    Coulomb, Repulsive, BV, BVV, Angle,
    clear_neighbor_cache,
)
from src.extensions.ewald import Ewald, clear_kvec_cache


CUTOFF = 6.0
ATOL   = 1e-10  # absolute tolerance on summary statistics


# ──────────────────────────────────────────────
# Frame fixture
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/vasprun.xml",
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
    # Values in eV: Coulomb now includes k_e = 14.3996454784 eV·Å (previously
    # returned e²/Å, i.e. these values / k_e).
    coul = Coulomb(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8}, cutoff=CUTOFF)
    e = coul.energy(frame.lattice, frame.species, frame.positions)
    f = coul.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(-3.665300372212974e+02, abs=ATOL)
    _check_array(f, 1.015927341747494e+01, 3.156162153161611e+00, -5.551115123125783e-15)


# ──────────────────────────────────────────────
# Repulsive
# ──────────────────────────────────────────────

def test_repulsive(frame):
    rep = Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=CUTOFF)
    e = rep.energy(frame.lattice, frame.species, frame.positions)
    f = rep.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(7.698420674672065e+00, abs=ATOL)
    # Functional form changed to the BVMD (B/r)^12 (was B/r^12, which made the
    # term ~B^12 ≈ 10^3–10^4 × too small); snapshot re-captured.
    # Forces fixed in commit "potentials: remove pair double-count in
    # Repulsive/Ewald-real forces"; previous (buggy) values were exactly 2×.
    # Then sign flipped in a follow-up: forces() returns f = -∂E/∂x.
    _check_array(f, 7.653934644391810e+00, 2.738716070115037e+00, 1.498801083243961e-15)


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
    _check_array(V, 1.589776941167630e+01, 4.056590571616593e+00, 9.550864764772406e+01)

def test_bv_energy_forces(frame):
    bv = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    e = bv.energy(frame.lattice, frame.species, frame.positions)
    f = bv.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(1.871272598225598e-01, abs=ATOL)
    # Updated after fixing the missing Newton 3rd-law term in BV.forces
    # (V_j-derivative contribution per pair); previous values were partial.
    # Force sum is now ~0 (machine epsilon) — the previous +1.00 reflected
    # the missing term.
    _check_array(f, 2.329191881834614e+00, 1.035814044402791e+00, 8.881784197001252e-16)


# ──────────────────────────────────────────────
# Bond Valence Vector
# ──────────────────────────────────────────────

_BVV_SP = {"Pb": {"W0": 0.5, "D": 0.1},
           "Ti": {"W0": 0.3, "D": 0.1},
           "O":  {"W0": 0.0, "D": 0.0}}

def test_bvv_W(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    W   = bvv.get_bvv(frame.lattice, frame.species, frame.positions)
    _check_array(W, 3.594970543350728e+00, 1.087795624782377e+00, 4.996003610813204e-16)

def test_bvv_energy_forces(frame):
    bvv = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF)
    e   = bvv.energy(frame.lattice, frame.species, frame.positions)
    f   = bvv.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(5.773241689600741e-02, abs=ATOL)
    # Updated after rewriting BVV.forces with the full 3×3 ∂(V r̂)/∂r_a
    # tensor (previously collapsed to a radial-only 3-vector and missing
    # both lateral and Newton 3rd-law components). Force sum is now ~0
    # (machine epsilon) — previously -2.24, reflecting the missing terms.
    _check_array(f, 8.671908294960492e-01, 3.570420271997536e-01, 8.326672684688674e-17)


# ──────────────────────────────────────────────
# Angle (O-centered, smaller cutoff so only nearest O-O-O triples)
# ──────────────────────────────────────────────

def test_angle(frame):
    ang = Angle(k=0.001, cutoff=3.5)
    e = ang.energy(frame.lattice, frame.species, frame.positions)
    f = ang.forces(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(1.540389977343968e+04, abs=ATOL)
    # Force sign flipped along with the global f = -∂E/∂x correction.
    _check_array(f, 2.317046426378549e+02, 6.753897187128561e+01, 3.552713678800501e-14)


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
