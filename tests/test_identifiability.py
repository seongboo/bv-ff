"""
Exact gauge symmetries of E_BV + E_BVV with free V0 (why V0 is held fixed).

E_BV  = Σ_i S_i (V_i - V0_i)²,  E_BVV = Σ_i D_i (W_i² - W0_i²)²,
V_ij = (r0_ij / r)^C_ij,  W_i = Σ_j V_ij r̂_ij.

(a) Rescaling. r0 → κ^{1/C} r0 multiplies every V_ij (hence V_i, W_i) by κ.
    With V0 → κ V0, S → S/κ², W0 → κ W0, D → D/κ⁴ each term is unchanged:
        S/κ² (κV - κV0)² = S (V - V0)²,  D/κ⁴ (κ²W² - κ²W0²)² = D (W² - W0²)².
(b) Sum rule. Each cation–O bond adds V_ij to both ends, so
    Σ_cations V_i = Σ_O V_i. The linear part -2 Σ S_i V0_i V_i is therefore
    unchanged (up to a constant) under S_O δV0_O = -S_c δV0_c for every cation c;
    the quadratic constant Σ S_i (V0_i² ...) shifts E by a configuration-
    independent amount, which the fitted energy reference absorbs.

Both leave all forces (and energy differences) exactly invariant, so V0 is not
identifiable from energies/forces. Fixing V0 (formal valences) breaks both.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parsers.dataset          import load_dataset, DatasetEntry
from parsers.controls_parser  import parse_controls
from src.potentials           import BV, BVV, BVFF, clear_neighbor_cache
from src.fitting              import free_mask


_ROOT = Path(__file__).resolve().parent.parent
_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.3}, "O": {"V0": 2.0, "S": 0.7}}
_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}
_VSP  = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.2}, "O": {"W0": 0.1, "D": 0.05}}


@pytest.fixture(scope="module")
def frames():
    clear_neighbor_cache()
    return load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=130, stride=10,
    )]).frames


def _model(sp, pp, vsp):
    return BVFF([BV(sp, pp, cutoff=6.0), BVV(vsp, pp, cutoff=6.0)])


def test_gauge_rescaling_is_exact(frames):
    k = 0.93
    pp2  = {p: {"r0": v["r0"] * k ** (1.0 / v["C"]), "C": v["C"]} for p, v in _PP.items()}
    sp2  = {s: {"V0": k * v["V0"], "S": v["S"] / k**2} for s, v in _SP.items()}
    vsp2 = {s: {"W0": k * v["W0"], "D": v["D"] / k**4} for s, v in _VSP.items()}
    m1, m2 = _model(_SP, _PP, _VSP), _model(sp2, pp2, vsp2)
    for fr in frames:
        e1, f1 = m1.energy_and_forces(fr.lattice, fr.species, fr.positions)
        e2, f2 = m2.energy_and_forces(fr.lattice, fr.species, fr.positions)
        assert e2 == pytest.approx(e1, rel=1e-10)
        np.testing.assert_allclose(f2, f1, rtol=1e-9, atol=1e-12)


def test_gauge_sum_rule_shift_is_exact_up_to_constant(frames):
    d_O = 0.25
    sp2 = {s: dict(v) for s, v in _SP.items()}
    sp2["O"]["V0"]  += d_O
    sp2["Pb"]["V0"] -= _SP["O"]["S"] * d_O / _SP["Pb"]["S"]
    sp2["Ti"]["V0"] -= _SP["O"]["S"] * d_O / _SP["Ti"]["S"]
    m1, m2 = _model(_SP, _PP, _VSP), _model(sp2, _PP, _VSP)
    diffs = []
    for fr in frames:
        e1, f1 = m1.energy_and_forces(fr.lattice, fr.species, fr.positions)
        e2, f2 = m2.energy_and_forces(fr.lattice, fr.species, fr.positions)
        diffs.append(e2 - e1)
        np.testing.assert_allclose(f2, f1, rtol=1e-9, atol=1e-12)
    assert np.ptp(diffs) < 1e-10, "energy shift must be configuration-independent"
    assert abs(diffs[0]) > 1e-3, "sanity: the constant itself is nonzero"


def test_v0_fixed_by_default():
    keys = ["coulomb.Pb", "BV.species.Pb.V0", "BV.species.Pb.S", "BV.species.O.V0", "BVV.O.W0"]
    fixed = parse_controls(str(_ROOT / "controls.toml")).fitting.fixed
    mask = free_mask(keys, fixed)
    assert mask.tolist() == [True, False, True, False, False]
    assert free_mask(keys, []).all()
