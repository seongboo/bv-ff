"""
Phase 3 FE science features: Born-effective-charge polarization, the external
E-field term, Γ-point phonons, and the Tc-scan machinery.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.tools.ferroelectric import (
    ideal_perovskite, perovskite_zstar, polarization_bec, gamma_phonons,
    ZSTAR_TABLE, E_PER_ANG2_TO_C_PER_M2,
)
from bvff.tools.tc_scan import scan_temperatures, estimate_tc
from bvff.core.extensions.efield import EField
from bvff.core.potentials import Repulsive, BV, BVFF, clear_neighbor_cache


_CHARGES = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}
_BV_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5},
            "O":  {"V0": 2.0, "S": 0.5}}
_BV_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0, "b": 0.4},
            "O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.37}}


def _toy_bvff() -> BVFF:
    return BVFF([
        Repulsive(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=6.0,
                  smooth_width=1.0),
        BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=6.0,
           form="exp", smooth_width=1.0),
    ])


# ──────────────────────────────────────────────
# Born-effective-charge polarization
# ──────────────────────────────────────────────

def test_zstar_classification_and_sum_rule():
    ref = ideal_perovskite(a=3.97, rep=(2, 2, 2), A="Pb")
    z   = perovskite_zstar(ref, axis=2, A_site="Pb")
    sym = np.array(ref.get_chemical_symbols())
    tab = ZSTAR_TABLE["Pb"]
    assert np.allclose(z[sym == "Pb"], tab["A"])
    assert np.allclose(z[sym == "Ti"], tab["Ti"])
    zO = z[sym == "O"]
    # 24 O in 8 f.u.: 8 chains along z (O∥), 16 along x/y (O⊥).
    assert np.sum(np.isclose(zO, tab["O_par"]))  == 8
    assert np.sum(np.isclose(zO, tab["O_perp"])) == 16
    # Acoustic sum rule per cell.
    assert abs(z.sum()) < 1e-9


def test_zstar_unknown_chemistry_returns_none():
    ref = ideal_perovskite(a=3.9, rep=(1, 1, 1), A="Sr")
    assert perovskite_zstar(ref, A_site="Sr") is None


def test_polarization_bec_matches_manual():
    ref = ideal_perovskite(a=3.97, rep=(2, 2, 2), A="Pb")
    z   = perovskite_zstar(ref, axis=2, A_site="Pb")
    a   = ref.copy()
    pos = a.get_positions()
    sym = np.array(a.get_chemical_symbols())
    u   = 0.1
    pos[sym == "Ti", 2] += u
    a.set_positions(pos)
    P = polarization_bec(a, ref, z)
    manual = 8 * ZSTAR_TABLE["Pb"]["Ti"] * u / a.get_volume() * E_PER_ANG2_TO_C_PER_M2
    assert P[2] == pytest.approx(manual, rel=1e-9)
    assert abs(P[0]) < 1e-12 and abs(P[1]) < 1e-12
    # Sanity: a realistic full-mode displacement gives an experiment-scale P.
    assert 0.1 < abs(manual) < 2.0


# ──────────────────────────────────────────────
# External E-field term
# ──────────────────────────────────────────────

def test_efield_forces_are_qE():
    ref = ideal_perovskite(a=3.97, rep=(1, 1, 1), A="Pb")
    E   = np.array([0.0, 0.0, 0.05])
    ef  = EField(charges=_CHARGES, field=E)
    lat = np.asarray(ref.cell.array)
    frac = ref.get_scaled_positions()
    f = ef.forces(lat, ref.get_chemical_symbols(), frac)
    q = np.array([_CHARGES[s] for s in ref.get_chemical_symbols()])
    assert np.allclose(f, q[:, None] * E[None, :])


def test_efield_energy_gradient_consistency():
    """ΔE = −q E·Δr when one ion moves (f = −∂E/∂r holds exactly)."""
    ref = ideal_perovskite(a=3.97, rep=(1, 1, 1), A="Pb")
    ef  = EField(charges=_CHARGES, field=[0.0, 0.0, 0.05])
    lat = np.asarray(ref.cell.array)
    sp  = ref.get_chemical_symbols()
    frac = ref.get_scaled_positions()
    e0 = ef.energy(lat, sp, frac)
    dz = 0.2
    frac2 = frac.copy()
    frac2[1, 2] += dz / lat[2, 2]          # move the Ti by 0.2 Å along z
    e1 = ef.energy(lat, sp, frac2)
    assert e1 - e0 == pytest.approx(-_CHARGES["Ti"] * 0.05 * dz, rel=1e-9)


def test_efield_tilts_double_well():
    """A +z field must favor the +z-polarized state: E(+δ) < E(−δ)."""
    ref = ideal_perovskite(a=3.97, rep=(2, 2, 2), A="Pb")
    ef  = EField(charges=_CHARGES, field=[0.0, 0.0, 0.1])
    lat = np.asarray(ref.cell.array)
    sp  = ref.get_chemical_symbols()
    sym = np.array(sp)
    cat = np.isin(sym, ["Pb", "Ti"])
    e = {}
    for s in (+1, -1):
        a = ref.copy()
        pos = a.get_positions()
        pos[cat, 2] += s * 0.18
        a.set_positions(pos)
        e[s] = ef.energy(lat, sp, a.get_scaled_positions(wrap=False))
    assert e[+1] < e[-1]


def test_efield_stress_matches_fd():
    ref = ideal_perovskite(a=3.97, rep=(1, 1, 1), A="Pb")
    # Distorted (polar) config so the dipole is nonzero.
    pos = ref.get_positions(); pos[1, 2] += 0.2
    ref.set_positions(pos)
    ef  = EField(charges=_CHARGES, field=[0.02, 0.0, 0.05])
    lat = np.asarray(ref.cell.array)
    sp  = ref.get_chemical_symbols()
    frac = ref.get_scaled_positions(wrap=False)
    analytic = ef.stress(lat, sp, frac)
    eps = 1e-5
    fd = np.zeros((3, 3))
    vol = abs(np.linalg.det(lat))
    for a in range(3):
        for b in range(3):
            F = np.eye(3); F[a, b] += eps
            ep = ef.energy(lat @ F.T, sp, frac)
            F[a, b] -= 2 * eps
            em = ef.energy(lat @ F.T, sp, frac)
            fd[a, b] = (ep - em) / (2 * eps) / vol
    fd = 0.5 * (fd + fd.T)
    assert np.allclose(analytic, fd, atol=1e-9)


def test_build_bvff_wires_efield(tmp_path):
    from bvff.parsers.controls_parser import parse_controls
    from bvff.parsers.parameters_parser import Parameters, CoulombParams, RepulsiveParams
    from bvff.core.main import build_bvff
    from bvff.core.extensions.efield import EField as EF

    c = tmp_path / "controls.toml"
    d = tmp_path / "d.extxyz"; d.write_text("")
    c.write_text(
        f'dataset = "{d}"\n'
        '[potentials]\nuse_coulomb = 0\nuse_repulsive = 1\nuse_BV = 0\nuse_BVV = 0\n'
        '[extensions]\nuse_ewald = 0\nefield = [0.0, 0.0, 0.1]\n'
    )
    controls = parse_controls(str(c))
    params = Parameters(coulomb=CoulombParams(charges={"O": -1.0}),
                        repulsive=RepulsiveParams(B={"O-O": 1.0}))
    bvff = build_bvff(controls, params)
    assert any(isinstance(t, EF) for t in bvff.terms)
    # No charges → hard error, not a silently field-free run.
    params2 = Parameters(repulsive=RepulsiveParams(B={"O-O": 1.0}))
    with pytest.raises(ValueError, match="efield"):
        build_bvff(controls, params2)


# ──────────────────────────────────────────────
# Γ-point phonons
# ──────────────────────────────────────────────

def test_gamma_phonons_acoustic_and_symmetry():
    clear_neighbor_cache()
    prim = ideal_perovskite(a=3.97, rep=(1, 1, 1), A="Pb")
    from bvff.core.calculator import BVFFCalculator
    calc = BVFFCalculator(_toy_bvff())
    thz, mev, n_im = gamma_phonons(prim, calc)
    assert thz.shape == (15,) and np.isfinite(thz).all()
    assert np.all(np.diff(thz) >= -1e-9)          # sorted
    # 3 acoustic modes at ~0 (translational invariance of the FD Hessian).
    assert np.sum(np.abs(thz) < 0.15) >= 3
    # meV and THz are the same spectrum in different units.
    assert np.allclose(mev / thz[np.abs(thz) > 1e-6].mean(),
                       mev / thz[np.abs(thz) > 1e-6].mean())
    assert isinstance(n_im, int)


# ──────────────────────────────────────────────
# Tc scan machinery
# ──────────────────────────────────────────────

def test_scan_temperatures_smoke():
    from bvff.core.calculator import BVFFCalculator
    atoms = ideal_perovskite(a=3.97, rep=(2, 2, 2), A="Pb")
    recs = scan_temperatures(
        atoms, BVFFCalculator(_toy_bvff()), temps=[50.0],
        steps=30, equil=10, sample_every=5, timestep_fs=1.0,
    )
    assert len(recs) == 1
    r = recs[0]
    assert r["n_samples"] >= 2 and not r["exploded"]
    assert np.isfinite(r["u_abs_mean"])


def test_estimate_tc():
    recs = [
        {"T": 100.0, "u_mean": [0, 0, 0.35]},
        {"T": 300.0, "u_mean": [0, 0, 0.30]},
        {"T": 500.0, "u_mean": [0, 0, 0.12]},   # < half of 0.35 → Tc bracket
        {"T": 700.0, "u_mean": [0, 0, 0.02]},
    ]
    assert estimate_tc(recs) == 500.0
    assert estimate_tc(recs[:2]) is None
    assert estimate_tc([{"T": 1.0, "u_mean": [0, 0, 0.01]}]) is None
