"""
Fitting-loss correctness: the energy-reference offset and block normalization
added to ``compute_loss``, and a smoke test that the seeded least-squares
optimizer actually reduces the loss.

These guard the methodology fixes:
  - energies are fit *up to a constant* (the classical and DFT energy scales
    differ by an un-fittable offset), so the loss must be invariant to a constant
    shift of the reference energies;
  - the energy/force blocks are normalized so they are commensurable;
  - the optimizer makes progress from the physical seed.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from bvff.parsers.dataset            import load_dataset, DatasetEntry
from bvff.parsers.parameters_parser  import (
    Parameters, CoulombParams, RepulsiveParams,
    BVParams, BVSpecies, BVPair, BVVParams, BVVSpecies,
)
from bvff.parsers.controls_parser    import PotentialControls
from bvff.core.potentials             import Coulomb, Repulsive, BV, BVV, BVFF, clear_neighbor_cache
from bvff.core.fitting                import (
    compute_loss, fit, reference_scales,
    bound_saturation_report, build_bounds, params_to_vector, ParamTransform,
)


CUTOFF = 6.0


def _params() -> Parameters:
    p = Parameters(cutoff=CUTOFF)
    p.coulomb   = CoulombParams(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8})
    p.repulsive = RepulsiveParams(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28})
    p.BV  = BVParams(
        species={"Pb": BVSpecies(2.0, 0.5), "Ti": BVSpecies(4.0, 0.5), "O": BVSpecies(2.0, 0.5)},
        pairs={"O-Pb": BVPair(2.112, 6.0, 0.37), "O-Ti": BVPair(1.815, 5.2, 0.37)},
    )
    p.BVV = BVVParams(
        species={"Pb": BVVSpecies(0.5, 0.1), "Ti": BVVSpecies(0.3, 0.1), "O": BVVSpecies(0.0, 0.0)},
        pairs={"O-Pb": BVPair(2.112, 6.0, 0.37), "O-Ti": BVPair(1.815, 5.2, 0.37)},
    )
    return p


def _builder(p: Parameters) -> BVFF:
    return BVFF([
        Coulomb(p.coulomb.charges, CUTOFF),
        Repulsive(p.repulsive.B, CUTOFF),
        BV({a: {"V0": s.V0, "S": s.S} for a, s in p.BV.species.items()},
           {k: {"r0": v.r0, "C": v.C, "b": v.b} for k, v in p.BV.pairs.items()},
           CUTOFF, form="exp"),
        BVV({a: {"W0": s.W0, "D": s.D} for a, s in p.BVV.species.items()},
            {k: {"r0": v.r0, "C": v.C, "b": v.b} for k, v in p.BVV.pairs.items()},
            CUTOFF, form="exp"),
    ])


_PC = PotentialControls(
    use_coulomb=True, use_repulsive=True, use_buckingham=False,
    use_BV=True, use_BVV=True, use_angle=False, bv_form="exp",
)


@pytest.fixture(scope="module")
def frames(pto_300k):
    clear_neighbor_cache()
    data = load_dataset(entries=[DatasetEntry(
        path=str(pto_300k),
        frame_start=100, frame_end=140, stride=5,
    )])
    return data.frames


def test_loss_invariant_to_energy_offset(frames):
    """A constant shift of all reference energies must not change the loss —
    energies are fit only up to a constant."""
    bvff   = _builder(_params())
    nE, nF, nS = reference_scales(frames)
    l0 = compute_loss(bvff, frames, 1, 1, 1, has_stress=False, norm_E=nE, norm_F=nF)

    shifted = [dataclasses.replace(f, energy=f.energy + 12345.6789) for f in frames]
    l1 = compute_loss(bvff, shifted, 1, 1, 1, has_stress=False, norm_E=nE, norm_F=nF)

    assert l1 == pytest.approx(l0, rel=1e-9, abs=1e-9)


def test_offset_flag_actually_matters(frames):
    """With offset removal off, a constant energy shift *does* change the loss —
    confirms the invariance above comes from the offset removal, not a no-op."""
    bvff   = _builder(_params())
    nE, nF, nS = reference_scales(frames)
    shifted = [dataclasses.replace(f, energy=f.energy + 5.0) for f in frames]
    l_no_off_0 = compute_loss(bvff, frames,  1, 1, 1, has_stress=False,
                              norm_E=nE, norm_F=nF, subtract_energy_offset=False)
    l_no_off_1 = compute_loss(bvff, shifted, 1, 1, 1, has_stress=False,
                              norm_E=nE, norm_F=nF, subtract_energy_offset=False)
    assert l_no_off_0 != pytest.approx(l_no_off_1)


def test_reference_scales_positive(frames):
    nE, nF, nS = reference_scales(frames)
    assert nE > 0 and nF > 0 and nS > 0


def test_normalization_changes_balance(frames):
    """Normalizing by the force scale must change the loss versus norm=1 (i.e.
    the normalization is actually applied)."""
    bvff = _builder(_params())
    raw  = compute_loss(bvff, frames, 1, 1, 1, has_stress=False, norm_E=1.0, norm_F=1.0)
    nE, nF, nS = reference_scales(frames)
    norm = compute_loss(bvff, frames, 1, 1, 1, has_stress=False, norm_E=nE, norm_F=nF)
    assert raw != pytest.approx(norm)


def test_least_squares_reduces_loss(frames):
    """The seeded trust-region optimizer must improve on the initial loss.
    Runs with the default log-space transform and checks the diagnostics dict
    is filled (bound flags, loss bookkeeping)."""
    params = _params()
    nE, nF, nS = reference_scales(frames)
    l0 = compute_loss(_builder(params), frames, 1, 1, 1, has_stress=False, norm_E=nE, norm_F=nF)
    diag: dict = {}
    _, best = fit(
        _builder, params, _PC, frames,
        w_E=1.0, w_F=1.0, w_S=1.0, has_stress=False, use_stress=False,
        maxiter=30, n_jobs=1, optimizer="least_squares",
        diagnostics=diag,
    )
    assert best < l0
    assert diag["optimizer"] == "least_squares"
    assert diag["n_parameters"] > 0
    assert diag["best_loss"] == pytest.approx(best)
    assert isinstance(diag["bound_flags"], list)


# ──────────────────────────────────────────────
# Bound-saturation diagnostics + log-space transform
# ──────────────────────────────────────────────

def test_bound_saturation_report_flags():
    """The exact pathology of the 2026-07 PbTiO3 fit must be flagged: values
    pinned at either bound (a ~0 value reports as collapse), while interior
    values pass silently."""
    keys   = ["repulsive.O-Ti", "repulsive.O-O", "BV.species.Ti.V0",
              "BVV.Pb.W0", "BV.pairs.O-Ti.r0"]
    bounds = [(0.0, 1e4), (0.0, 1e4), (0.1, 10.0), (0.0, 5.0), (0.5, 3.0)]
    x      = np.array([6.3e-32, 9999.999999999998, 10.0, 2.2e-5, 1.9])

    flags = {f["key"]: f["flag"] for f in bound_saturation_report(x, keys, bounds)}
    assert flags["repulsive.O-Ti"]   == "collapsed_to_zero"
    assert flags["repulsive.O-O"]    == "at_upper_bound"
    assert flags["BV.species.Ti.V0"] == "at_upper_bound"
    assert flags["BVV.Pb.W0"]        == "at_lower_bound"
    assert "BV.pairs.O-Ti.r0" not in flags


def test_bound_saturation_report_log_space_criteria():
    """With the log transform, saturation is judged in the space the optimizer
    searched: a small-but-legitimate magnitude (B=0.8, 8000x above the floor)
    must NOT be flagged, while a walk to the effective-zero floor must be."""
    keys   = ["repulsive.O-Pb", "repulsive.O-Ti"]
    bounds = [(0.0, 1e4), (0.0, 1e4)]
    tr     = ParamTransform(keys, bounds, log_space=True)
    floor  = tr.lo_eff[1]                      # 1e-8 * 1e4 = 1e-4

    flags = {f["key"]: f["flag"] for f in bound_saturation_report(
        np.array([0.8, floor]), keys, bounds, transform=tr)}
    assert "repulsive.O-Pb" not in flags       # linear-span tol would false-flag this
    assert flags["repulsive.O-Ti"] == "at_lower_bound"


def test_param_transform_roundtrip_and_bounds():
    """Log-space transform: magnitude params only, exact roundtrip for
    positive seeds, zero seeds clipped up to the positive floor, seed always
    inside the optimizer-space bounds."""
    params   = _params()
    x0, keys = params_to_vector(params, _PC, None, fit_charges=False)
    bounds   = build_bounds(keys)
    tr       = ParamTransform(keys, bounds, log_space=True)

    # Magnitude params (repulsive B, BV S, BVV D) are log; shape params are not.
    log_keys = {k for k, m in zip(keys, tr.is_log) if m}
    assert "repulsive.O-O" in log_keys
    assert "BV.species.Ti.S" in log_keys
    assert "BVV.Ti.D" in log_keys
    assert "BV.pairs.O-Ti.r0" not in log_keys
    assert "BVV.Pb.W0" not in log_keys

    z  = tr.to_opt(x0)
    x1 = tr.from_opt(z)
    pos = x0 > 0
    np.testing.assert_allclose(x1[pos], x0[pos], rtol=1e-12)
    zero_log = (x0 == 0) & tr.is_log          # e.g. BVV O D = 0.0 seed
    assert np.all(x1[zero_log] > 0)           # clipped up to the floor, not -inf
    zero_lin = (x0 == 0) & ~tr.is_log         # e.g. BVV O W0 = 0.0 seed
    np.testing.assert_allclose(x1[zero_lin], 0.0)

    zb  = tr.opt_bounds()
    zlb = np.array([b[0] for b in zb])
    zub = np.array([b[1] for b in zb])
    assert np.all(zlb < zub)
    assert np.all(z >= zlb - 1e-12) and np.all(z <= zub + 1e-12)
    # A log param at its optimizer-space lower bound maps to a physically
    # negligible (but positive, finite) value.
    i = keys.index("repulsive.O-O")
    assert 0.0 < 10.0 ** zlb[i] < 1e-3


def test_param_transform_identity_when_disabled():
    params   = _params()
    x0, keys = params_to_vector(params, _PC, None, fit_charges=False)
    bounds   = build_bounds(keys)
    tr       = ParamTransform(keys, bounds, log_space=False)
    assert not tr.is_log.any()
    np.testing.assert_allclose(tr.to_opt(x0), x0)
    np.testing.assert_allclose(tr.from_opt(x0), x0)
    assert tr.opt_bounds() == [(float(lo), float(hi)) for lo, hi in bounds]
