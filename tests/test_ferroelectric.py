"""
Ferroelectric tooling (bvff/tools/ferroelectric.py): reference structure,
double-well scan, point-charge polarization, relaxation, and a short NVT MD.

These check the *machinery* (shapes, invariants, that ASE drivers run), not the
physics quality of any particular fit.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.dataset   import Frame
from bvff.core.potentials    import Coulomb, Repulsive, BV, BVV, BVFF
from bvff.core.calculator    import BVFFCalculator
from bvff.tools.ferroelectric import (
    ideal_perovskite, double_well_scan, well_depth, polarization, relax, run_nvt,
    cation_offcentering, run_validation,
)


CUTOFF = 6.0
_CHARGES = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}
_B       = {"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}
_BV_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BV_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}


def _calc():
    return BVFFCalculator(BVFF([
        Coulomb(_CHARGES, CUTOFF), Repulsive(_B, CUTOFF), BV(_BV_SP, _BV_PP, CUTOFF),
    ]))


def test_ideal_perovskite_composition():
    at = ideal_perovskite(a=3.9, rep=(2, 2, 2))
    sym = at.get_chemical_symbols()
    assert len(at) == 40
    assert sym.count("Pb") == 8 and sym.count("Ti") == 8 and sym.count("O") == 24


def test_polarization_of_reference_is_zero():
    ref = ideal_perovskite(a=3.9)
    P = polarization(ref, _CHARGES, ref)            # zero displacement → zero P
    np.testing.assert_allclose(P, 0.0, atol=1e-12)


def test_polarization_sign_and_scale():
    """Displacing cations +z must give +z polarization of a sensible magnitude."""
    ref = ideal_perovskite(a=3.9)
    a = ref.copy()
    pos = a.get_positions()
    sym = np.array(a.get_chemical_symbols())
    pos[np.isin(sym, ["Ti", "Pb"]), 2] += 0.1
    a.set_positions(pos)
    P = polarization(a, _CHARGES, ref)
    assert P[2] > 0 and abs(P[0]) < 1e-9 and abs(P[1]) < 1e-9
    assert 0.01 < abs(P[2]) < 5.0                    # plausible C/m² range


def test_cation_offcentering_zero_on_primitive_cell():
    """Regression: with only the single nearest image per O, the 5-atom
    primitive cell (3 O atoms < 6 octahedral neighbors) reported a spurious
    |u| = a*sqrt(3)/6 for a perfectly centrosymmetric structure. The image
    expansion must make it exactly 0."""
    prim = ideal_perovskite(a=3.97, rep=(1, 1, 1))
    assert np.linalg.norm(cation_offcentering(prim, "Ti")) < 1e-9


def test_cation_offcentering_distinguishes_polar_from_cubic():
    ref = ideal_perovskite(a=3.9)
    assert np.linalg.norm(cation_offcentering(ref, "Ti")) < 1e-9   # centrosymmetric → 0
    a = ref.copy()
    pos = a.get_positions()
    sym = np.array(a.get_chemical_symbols())
    pos[sym == "Ti", 2] += 0.2                                      # polar shift of Ti
    a.set_positions(pos)
    u = cation_offcentering(a, "Ti")
    assert u[2] == pytest.approx(0.2, abs=1e-6) and abs(u[0]) < 1e-9


def test_double_well_scan_shape_and_zero():
    ref = ideal_perovskite(a=3.9)
    amps, dE = double_well_scan(ref, _calc(), amplitudes=np.linspace(-0.3, 0.3, 13))
    assert amps.shape == dE.shape == (13,)
    assert dE[np.argmin(np.abs(amps))] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(dE).all()
    depth, dmin = well_depth(amps, dE)
    assert np.isfinite(depth) and np.isfinite(dmin)


def test_relax_reduces_force():
    ref = ideal_perovskite(a=3.9)
    a = ref.copy()
    pos = a.get_positions()
    pos[1, 2] += 0.15                                # nudge one Ti
    a.set_positions(pos)
    calc = _calc()
    a.calc = calc
    fmax0 = np.abs(a.get_forces()).max()
    relax(a, calc, fmax=0.05, steps=200)
    fmax1 = np.abs(a.get_forces()).max()
    assert fmax1 < fmax0


def test_nvt_md_runs_and_is_finite():
    ref = ideal_perovskite(a=3.9)
    e = run_nvt(ref.copy(), _calc(), temperature_K=50.0, steps=20,
                timestep_fs=0.5, seed=0)
    assert e.size > 0 and np.isfinite(e).all()


def test_polarization_invariant_to_pbc_wrap():
    """An atom translated by a full lattice vector (PBC wrap, routine after
    relax/MD) must not change P — the minimum-image displacement removes the
    spurious ~q*L/V branch jump the raw position difference would add."""
    ref = ideal_perovskite(a=3.9)
    a = ref.copy()
    pos = a.get_positions()
    sym = np.array(a.get_chemical_symbols())
    pos[np.isin(sym, ["Ti", "Pb"]), 2] += 0.1
    a.set_positions(pos)
    P0 = polarization(a, _CHARGES, ref)

    wrapped = a.copy()
    pos = wrapped.get_positions()
    pos[0] += wrapped.cell.array[2]          # translate one atom by +c (a wrap)
    pos[16] -= wrapped.cell.array[0]         # and one O by -a
    wrapped.set_positions(pos)
    P1 = polarization(wrapped, _CHARGES, ref)

    np.testing.assert_allclose(P1, P0, atol=1e-10)


def _frame_from_atoms(atoms) -> Frame:
    lattice = np.asarray(atoms.cell.array, dtype=float)
    return Frame(
        index=0, lattice=lattice, species=atoms.get_chemical_symbols(),
        positions=atoms.get_scaled_positions(wrap=False),
        energy=0.0, forces=np.zeros((len(atoms), 3)), stress=None,
    )


def test_run_validation_persists_artifacts(tmp_path):
    """run_validation must return the verdict dict AND write it to
    ferroelectric_validation.toml — the whole point is an auditable record."""
    import tomllib

    ref = ideal_perovskite(a=3.9)
    polar = ref.copy()
    pos = polar.get_positions()
    sym = np.array(polar.get_chemical_symbols())
    pos[np.isin(sym, ["Ti", "Pb"]), 2] += 0.1
    polar.set_positions(pos)

    result = run_validation(
        _calc(), tmp_path,
        charges     = _CHARGES,
        data_frame  = _frame_from_atoms(polar),
        amplitudes  = np.linspace(-0.2, 0.2, 5),
        relax_steps = 5,
        nvt_steps   = 10,
    )

    assert isinstance(result["passed"], bool)
    assert result["double_well"]["classification"] in (
        "paraelectric", "collapse", "double_well")
    assert result["retention"]["tested"] is True
    assert np.isfinite(result["nvt"]["max_abs_drift_eV"])

    toml_path = tmp_path / "ferroelectric_validation.toml"
    assert toml_path.exists()
    with open(toml_path, "rb") as f:
        persisted = tomllib.load(f)
    assert persisted["passed"] == result["passed"]
    assert persisted["double_well"]["classification"] == \
        result["double_well"]["classification"]
    assert len(persisted["double_well"]["dE_meV_per_fu"]) == 5
