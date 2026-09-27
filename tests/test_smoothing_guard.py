"""
Phase 1 MD-conservation tests: cutoff smoothing, the Buckingham inner guard,
and the Ewald neutralizing-background term.

Smoothing contract: with smooth_width > 0 every short-range term's energy and
forces go continuously to zero at the cutoff (no φ(r_c) jump when a pair
crosses), and forces remain the exact gradient of the smoothed energy.

Guard contract: Buckingham stays bounded (no −∞ catastrophe) below the wall,
joins the raw potential continuously in energy AND force at the switch point,
and warns loudly when the fitted parameters provide no repulsive wall at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials import (
    Coulomb, Repulsive, Buckingham, BV, BVV, Angle,
    _switch, clear_neighbor_cache,
)
from src.extensions.ewald import Ewald, ewald_alpha, clear_kvec_cache


CUTOFF = 6.0
WIDTH  = 1.0

_BV_SP  = {"Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BV_PP  = {"O-Ti": {"r0": 1.81, "C": 5.2}}
_BVV_SP = {"Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.0, "D": 0.0}}
_BUCK   = {"O-Ti": {"A": 700.0, "rho": 0.35, "C": 30.0}}


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/pbtio3_222_300K.parquet",
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    return data.frames[0]


def _dimer(r: float, species=("O", "Ti"), box: float = 15.0):
    """Two atoms r apart along x in a box large enough that only the direct
    pair is inside the cutoff (box - r > CUTOFF for all r used here)."""
    lattice   = np.eye(3) * box
    positions = np.array([[0.0, 0.0, 0.0], [r / box, 0.0, 0.0]])
    return lattice, list(species), positions


def _smoothed_terms():
    return [
        Coulomb(charges={"Ti": 1.0, "O": -1.0}, cutoff=CUTOFF, smooth_width=WIDTH),
        Repulsive(B={"O-Ti": 1.28}, cutoff=CUTOFF, smooth_width=WIDTH),
        Buckingham(params=_BUCK, cutoff=CUTOFF, smooth_width=WIDTH),
        BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
           form="exp", smooth_width=WIDTH),
        BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
            form="exp", smooth_width=WIDTH),
    ]


# ──────────────────────────────────────────────
# Switch function
# ──────────────────────────────────────────────

def test_switch_endpoints_and_derivative():
    r = np.linspace(CUTOFF - WIDTH - 0.5, CUTOFF + 0.2, 400)
    S, dS = _switch(r, CUTOFF, WIDTH)
    # Identity below the taper window, zero at/after the cutoff.
    assert np.all(S[r <= CUTOFF - WIDTH] == 1.0)
    assert np.all(dS[r <= CUTOFF - WIDTH] == 0.0)
    assert np.all(S[r >= CUTOFF] == 0.0)
    # dS/dr matches an FD derivative of S (C¹ everywhere, incl. both joins).
    h = 1e-6
    Sp, _ = _switch(r + h, CUTOFF, WIDTH)
    Sm, _ = _switch(r - h, CUTOFF, WIDTH)
    assert np.allclose((Sp - Sm) / (2 * h), dS, atol=1e-6)


def test_switch_disabled_is_truncation():
    r = np.linspace(0.5, CUTOFF, 50)
    S, dS = _switch(r, CUTOFF, 0.0)
    assert np.all(S == 1.0) and np.all(dS == 0.0)


# ──────────────────────────────────────────────
# Energy/force continuity at the cutoff
# ──────────────────────────────────────────────

@pytest.mark.parametrize("term_idx", range(5))
def test_energy_continuous_at_cutoff(term_idx):
    """E(r_c−δ) ≈ E(r_c+δ) ≈ E-baseline: no φ(r_c) jump with the taper on.
    (Untapered, e.g. Coulomb would jump by KE·q²/r_c ≈ 2.4 eV here.)"""
    clear_neighbor_cache()
    pot = _smoothed_terms()[term_idx]
    delta = 1e-5
    lat, sp, pos_in  = _dimer(CUTOFF - delta)
    _,   _,  pos_out = _dimer(CUTOFF + delta)
    e_in  = pot.energy(lat, sp, pos_in)
    e_out = pot.energy(lat, sp, pos_out)
    # Outside the cutoff a pair term is exactly its isolated-atom baseline.
    e_base = e_out
    assert abs(e_in - e_base) < 1e-10, type(pot).__name__

    # Forces also vanish continuously at the cutoff.
    f_in = pot.forces(lat, sp, pos_in)
    assert np.abs(f_in).max() < 1e-8, type(pot).__name__


@pytest.mark.parametrize("term_idx", range(5))
def test_forces_match_fd_with_taper(term_idx, frame):
    """f = -dE/dx must hold THROUGH the taper window (the S'(r) chain-rule
    term): checked by central differences inside the window (r ≈ 5.5 Å)."""
    clear_neighbor_cache()
    pot = _smoothed_terms()[term_idx]
    lat, sp, pos = _dimer(5.5)   # mid-taper
    f = pot.forces(lat, sp, pos)
    h = 1e-6
    for atom in (0, 1):
        for ax in range(3):
            dp = pos.copy(); dp[atom, ax] += h / lat[ax, ax]
            dm = pos.copy(); dm[atom, ax] -= h / lat[ax, ax]
            fd = -(pot.energy(lat, sp, dp) - pot.energy(lat, sp, dm)) / (2 * h)
            assert f[atom, ax] == pytest.approx(fd, abs=5e-6), \
                f"{type(pot).__name__} atom {atom} axis {ax}"


def test_fd_forces_full_frame_smoothed(frame):
    """Same gradient check on the real 40-atom PbTiO3 frame (all terms +
    Angle), a few random atom/axis probes each."""
    clear_neighbor_cache()
    terms = _smoothed_terms() + [Angle(k=0.001, cutoff=3.5, smooth_width=0.5)]
    # Use the full species map for BV/Buckingham terms on this frame.
    terms[3] = BV(
        species_params={"Pb": {"V0": 2.0, "S": 0.5}, **_BV_SP},
        pair_params={"O-Pb": {"r0": 2.06, "C": 6.0}, **_BV_PP},
        cutoff=CUTOFF, form="exp", smooth_width=WIDTH,
    )
    h = 1e-5
    lat = frame.lattice
    inv_lat = np.linalg.inv(lat)
    for pot in terms:
        f = pot.forces(lat, frame.species, frame.positions)
        for atom, ax in [(0, 0), (8, 1), (17, 2)]:
            dcart = np.zeros(3); dcart[ax] = h
            dfrac = np.zeros_like(frame.positions)
            dfrac[atom] = dcart @ inv_lat
            fd = -(
                pot.energy(lat, frame.species, frame.positions + dfrac)
                - pot.energy(lat, frame.species, frame.positions - dfrac)
            ) / (2 * h)
            assert f[atom, ax] == pytest.approx(fd, abs=2e-4), \
                f"{type(pot).__name__} atom {atom} axis {ax}"


# ──────────────────────────────────────────────
# Buckingham inner guard
# ──────────────────────────────────────────────

def test_guard_bounded_and_repulsive_inside():
    """Below the wall the guarded φ keeps rising inward (constant repulsive
    force) instead of plunging to −∞: no fusion pathway."""
    clear_neighbor_cache()
    buck = Buckingham(params=_BUCK, cutoff=CUTOFF)
    r_sw, phi_sw, dphi_sw = buck._guards["O-Ti"]
    assert 0.02 < r_sw < 2.0
    assert dphi_sw < 0.0     # repulsive at the switch point

    e_prev = None
    for r in (0.05, 0.2, 0.5, min(0.9 * r_sw, r_sw - 0.01)):
        lat, sp, pos = _dimer(r)
        e = buck.energy(lat, sp, pos)
        assert np.isfinite(e)
        if e_prev is not None:
            assert e_prev > e   # φ decreases monotonically outward in the guard
        e_prev = e
        # Force on atom 0 points away from atom 1 (repulsive).
        f = buck.forces(lat, sp, pos)
        assert f[0, 0] < 0.0 and f[1, 0] > 0.0

    # Raw (unguarded) Buckingham at 0.05 Å would be catastrophically negative.
    A, rho, C = _BUCK["O-Ti"]["A"], _BUCK["O-Ti"]["rho"], _BUCK["O-Ti"]["C"]
    assert A * np.exp(-0.05 / rho) - C / 0.05**6 < -1e8


def test_guard_joins_continuously():
    """Energy and force are continuous at r_sw (tangent-line construction)."""
    clear_neighbor_cache()
    buck = Buckingham(params=_BUCK, cutoff=CUTOFF)
    r_sw, _, dphi_sw = buck._guards["O-Ti"]
    delta = 1e-6
    lat, sp, pos_in  = _dimer(r_sw - delta)
    _,   _,  pos_out = _dimer(r_sw + delta)
    e_in, e_out = buck.energy(lat, sp, pos_in), buck.energy(lat, sp, pos_out)
    assert abs(e_in - e_out) < 1e-4 * abs(dphi_sw)   # ~|φ'|·2δ
    f_in  = buck.forces(lat, sp, pos_in)[0, 0]
    f_out = buck.forces(lat, sp, pos_out)[0, 0]
    assert f_in == pytest.approx(f_out, rel=1e-3)


def test_guard_no_wall_warns_and_stays_finite():
    """A ≈ 0 with C > 0 is purely attractive — no wall to cap at. The guard
    must warn (this fit cannot run MD) yet still bound the energy."""
    clear_neighbor_cache()
    import src.potentials as P
    P._BUCK_GUARD_WARNED.clear()
    with pytest.warns(UserWarning, match="no repulsive wall"):
        buck = Buckingham(params={"O-Ti": {"A": 1e-6, "rho": 0.3, "C": 50.0}},
                          cutoff=CUTOFF)
    lat, sp, pos = _dimer(0.05)
    assert np.isfinite(buck.energy(lat, sp, pos))
    assert np.isfinite(buck.forces(lat, sp, pos)).all()


def test_guard_skipped_without_dispersion():
    """C = 0 (pure Born-Mayer) is bounded and repulsive — no guard needed."""
    buck = Buckingham(params={"O-Ti": {"A": 700.0, "rho": 0.35, "C": 0.0}},
                      cutoff=CUTOFF)
    assert buck._guards["O-Ti"] is None


# ──────────────────────────────────────────────
# Ewald: background term + tinfoil default
# ──────────────────────────────────────────────

def test_ewald_epsilon_defaults_to_tinfoil():
    ew = Ewald(charges={"O": -1.0}, alpha=0.3, kmax=5, cutoff=CUTOFF)
    assert np.isinf(ew.epsilon)


def test_ewald_charged_cell_alpha_independent():
    """For a non-neutral cell the jellium background makes the total energy
    independent of the arbitrary splitting alpha; without it the energies
    would differ by KE·π·Q²/(2V)·(1/α₁² − 1/α₂²)."""
    lat = np.eye(3) * 8.0
    sp  = ["Na"]
    pos = np.zeros((1, 3))
    Q   = 3.0
    a1  = ewald_alpha(6.0)
    a2  = 1.4 * a1
    clear_kvec_cache(); clear_neighbor_cache()
    e1 = Ewald({"Na": Q}, alpha=a1, kmax=None, cutoff=6.0).energy(lat, sp, pos)
    clear_kvec_cache()
    e2 = Ewald({"Na": Q}, alpha=a2, kmax=None, cutoff=6.0).energy(lat, sp, pos)
    # Sanity: the drift the background cancels dwarfs the residual mismatch.
    uncorrected_gap = Ewald.KE * np.pi * Q**2 / (2 * 8.0**3) * abs(1 / a1**2 - 1 / a2**2)
    assert uncorrected_gap > 0.3
    assert abs(e1 - e2) < 0.01 * uncorrected_gap


def test_ewald_background_zero_for_neutral(frame):
    """The background term must not change neutral-cell energies."""
    q = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}   # 8·1.4 + 8·1.0 − 24·0.8 = 0
    ew = Ewald(q, alpha=0.3, kmax=5, cutoff=CUTOFF)
    qv = ew._get_charges(frame.species)
    assert abs(qv.sum()) < 1e-12
    # Q enters squared, so a 1e-15 neutrality residue gives an O(1e-30) eV
    # term — physically zero.
    assert abs(ew._energy_background(qv, 500.0)) < 1e-20


# ──────────────────────────────────────────────
# Angle triplet species filter
# ──────────────────────────────────────────────

def test_angle_ignores_non_oxygen_legs():
    """O center with one O and one Ti neighbor: the old center-only filter
    produced an O-centered O-?-Ti triple (penalizing a 90° geometry toward
    180°); with both legs filtered there is no triple at all."""
    clear_neighbor_cache()
    lat = np.eye(3) * 12.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0 / 12, 0.0, 0.0], [0.0, 2.0 / 12, 0.0]])
    ang = Angle(k=0.001, cutoff=3.5)
    assert ang.energy(lat, ["O", "O", "Ti"], pos) == 0.0
    assert np.all(ang.forces(lat, ["O", "O", "Ti"], pos) == 0.0)
    # Same geometry, all-O: triples exist and the 90°/45° bends cost energy.
    e_ooo = ang.energy(lat, ["O", "O", "O"], pos)
    assert e_ooo > 0.0
