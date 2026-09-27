"""
Regression tests for the exp-form phantom-bond bug (found 2026-07-05 by the
LAMMPS export cross-validation).

Unparameterized pairs used to fall back to r0=0, C=1, b=1 without a mask.
For the power form that yields (0/r)^C = 0 — silently correct. For the exp
form it yields exp(−r): a PHANTOM BOND between every unparameterized species
pair (Pb-Pb, Pb-Ti, Ti-Ti, O-O, ...), contaminating every valence sum the
moment all species carry S > 0 (i.e., every real fit).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.potentials import BV, BVV, clear_neighbor_cache


def _dimer(r: float, species, box: float = 15.0):
    lattice   = np.eye(3) * box
    positions = np.array([[0.0, 0.0, 0.0], [r / box, 0.0, 0.0]])
    return lattice, list(species), positions


@pytest.mark.parametrize("form", ["exp", "power"])
def test_unparameterized_pair_contributes_nothing_bv(form):
    """Ti-Ti has no BV pair: the valence sum must be exactly zero even though
    both atoms have S > 0 (the old exp-form fallback gave V = e^(−r) ≈ 0.05
    at 3 Å — enough to shift every fitted V0)."""
    clear_neighbor_cache()
    bv = BV(species_params={"Ti": {"V0": 4.0, "S": 0.7}},
            pair_params={"O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.37}},
            cutoff=6.0, form=form)
    lat, sp, pos = _dimer(3.0, ("Ti", "Ti"))
    V = bv.get_valence(lat, sp, pos)
    assert np.all(V == 0.0)
    # Energy is the pure V=0 offset; forces are exactly zero.
    e = bv.energy(lat, sp, pos)
    assert e == pytest.approx(2 * 0.7 * 4.0**2, rel=1e-12)
    assert np.all(bv.forces(lat, sp, pos) == 0.0)


@pytest.mark.parametrize("form", ["exp", "power"])
def test_unparameterized_pair_contributes_nothing_bvv(form):
    clear_neighbor_cache()
    bvv = BVV(species_params={"Ti": {"W0": 0.3, "D": 0.2}},
              pair_params={"O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.37}},
              cutoff=6.0, form=form)
    lat, sp, pos = _dimer(3.0, ("Ti", "Ti"))
    W = bvv.get_bvv(lat, sp, pos)
    assert np.all(W == 0.0)
    assert np.all(bvv.forces(lat, sp, pos) == 0.0)


def test_parameterized_pair_unaffected_by_mask():
    """The mask must not touch parameterized pairs: O-Ti dimer valence equals
    the closed form."""
    clear_neighbor_cache()
    bv = BV(species_params={"Ti": {"V0": 4.0, "S": 0.7}, "O": {"V0": 2.0, "S": 0.4}},
            pair_params={"O-Ti": {"r0": 1.9128352769026349, "C": 5.2,
                                  "b": 0.389853822040997}},
            cutoff=6.0, form="exp", smooth_width=1.0)
    lat, sp, pos = _dimer(2.0, ("O", "Ti"))
    V = bv.get_valence(lat, sp, pos)
    expected = np.exp((1.9128352769026349 - 2.0) / 0.389853822040997)
    assert V[0] == pytest.approx(expected, rel=1e-12)
    assert V[1] == pytest.approx(expected, rel=1e-12)
