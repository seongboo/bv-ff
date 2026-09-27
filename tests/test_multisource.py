"""
Multi-source fitting: per-ref_group energy offset, per-channel/per-frame weights,
and that compute_loss (anneal path) and _residuals (least_squares path) minimize
the SAME objective.
"""
from __future__ import annotations

import dataclasses
from collections import Counter

import numpy as np
import pytest

from bvff.parsers.dataset           import load_dataset, DatasetEntry
from bvff.parsers.parameters_parser import (
    Parameters, CoulombParams, RepulsiveParams, BVParams, BVSpecies, BVPair,
)
from bvff.parsers.controls_parser   import PotentialControls
from bvff.core.potentials            import Coulomb, Repulsive, BV, BVFF, clear_neighbor_cache
from bvff.core.fitting               import (
    compute_loss, _residuals, reference_scales, build_frame_weights,
    params_to_vector,
)


CUTOFF = 6.0
_PC = PotentialControls(use_coulomb=True, use_repulsive=True, use_buckingham=False,
                        use_BV=True, use_BVV=False, use_angle=False, bv_form="exp")


def _params():
    p = Parameters(cutoff=CUTOFF)
    p.coulomb   = CoulombParams(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8})
    p.repulsive = RepulsiveParams(B={"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28})
    p.BV = BVParams(
        species={"Pb": BVSpecies(2.0, 0.5), "Ti": BVSpecies(4.0, 0.5), "O": BVSpecies(2.0, 0.5)},
        pairs={"O-Pb": BVPair(2.112, 6.0, 0.37), "O-Ti": BVPair(1.815, 5.2, 0.37)},
    )
    return p


def _builder(p):
    return BVFF([
        Coulomb(p.coulomb.charges, CUTOFF),
        Repulsive(p.repulsive.B, CUTOFF),
        BV({a: {"V0": s.V0, "S": s.S} for a, s in p.BV.species.items()},
           {k: {"r0": v.r0, "C": v.C, "b": v.b} for k, v in p.BV.pairs.items()},
           CUTOFF, form="exp"),
    ])


@pytest.fixture(scope="module")
def frames(pto_300k):
    clear_neighbor_cache()
    d = load_dataset(entries=[DatasetEntry(
        path=str(pto_300k),
        frame_start=100, frame_end=160, stride=5)])
    return d.frames


def _assign(frames, groups, wE=1.0, wF=1.0):
    """Return copies of frames with ref_group / weights assigned."""
    out = []
    for fr, g in zip(frames, groups):
        out.append(dataclasses.replace(fr, ref_group=g, weight_E=wE, weight_F=wF))
    return out


def test_per_group_offset_invariance(frames):
    """A constant energy shift applied to one ref_group must not change the loss."""
    n = len(frames)
    grp = ["a"] * (n // 2) + ["b"] * (n - n // 2)
    fa = _assign(frames, grp)
    bvff = _builder(_params())
    gid, wE, wF, wS, _ = build_frame_weights(fa, "count")
    nE, nF, nS = reference_scales(fa, wE, gid)
    kw = dict(has_stress=False, norm_E=nE, norm_F=nF, wE=wE, wF=wF, wS=wS, gid=gid)
    l0 = compute_loss(bvff, fa, 1, 1, 1, **kw)
    # shift only group 'b' energies by a large constant
    fb = [dataclasses.replace(fr, energy=fr.energy + (777.0 if fr.ref_group == "b" else 0.0)) for fr in fa]
    l1 = compute_loss(bvff, fb, 1, 1, 1, **kw)
    assert l1 == pytest.approx(l0, rel=1e-9, abs=1e-9)


def test_single_frame_group_energy_absorbed(frames):
    """A 1-frame ref_group's energy is fully absorbed by its offset → its energy
    value cannot change the loss (forces still count)."""
    grp = ["anchor"] + ["aimd"] * (len(frames) - 1)
    fa = _assign(frames, grp)
    bvff = _builder(_params())
    gid, wE, wF, wS, _ = build_frame_weights(fa, "count")
    nE, nF, nS = reference_scales(fa, wE, gid)
    kw = dict(has_stress=False, norm_E=nE, norm_F=nF, wE=wE, wF=wF, wS=wS, gid=gid,
              w_F=0.0, w_S=0.0)
    # energy-only loss; perturb the lone anchor frame's energy arbitrarily
    l0 = compute_loss(bvff, fa, w_E=1.0, **kw)
    fb = [dataclasses.replace(fr, energy=fr.energy + (999.0 if fr.ref_group == "anchor" else 0.0)) for fr in fa]
    l1 = compute_loss(bvff, fb, w_E=1.0, **kw)
    assert l1 == pytest.approx(l0, abs=1e-12)


def test_count_norm_weight_shares(frames):
    """count-norm: per-frame weight = weight_X / N_ref_group (group-total share)."""
    grp = ["aimd"] * 8 + ["dft"] * 4
    fa = _assign(frames[:12], grp)
    for fr in fa:
        fr.weight_E = 1.0 if fr.ref_group == "aimd" else 3.0
    gid, wE, wF, wS, groups = build_frame_weights(fa, "count")
    # each aimd frame: 1/8 ; each dft frame: 3/4 ; group totals 1 and 3
    aimd = wE[np.array([f.ref_group for f in fa]) == "aimd"]
    dft  = wE[np.array([f.ref_group for f in fa]) == "dft"]
    assert np.allclose(aimd, 1.0 / 8) and np.allclose(dft, 3.0 / 4)
    assert aimd.sum() == pytest.approx(1.0) and dft.sum() == pytest.approx(3.0)


def _residual_consistency(frames, w_E, w_F):
    """½‖_residuals‖² must equal compute_loss² for an energy-only or force-only
    block (so each block's scaling matches the per-frame-mean RMSE)."""
    bvff_p = _params()
    bvff   = _builder(bvff_p)
    gid, wE, wF, wS, _ = build_frame_weights(frames, "count")
    nE, nF, nS = reference_scales(frames, wE, gid)
    x0, keys = params_to_vector(bvff_p, _PC, neutral_dep=None, fit_charges=False)
    counts = dict(Counter(frames[0].species))
    loss = compute_loss(bvff, frames, w_E, w_F, 0.0, has_stress=False,
                        norm_E=nE, norm_F=nF, wE=wE, wF=wF, wS=wS, gid=gid)
    res = _residuals(x0, keys, bvff_p, None, counts, _builder, frames,
                     w_E, w_F, nE, nF, n_jobs=1, wE=wE, wF=wF, gid=gid)
    return float(np.sum(res ** 2)), loss ** 2


def test_residuals_match_compute_loss_energy_block(frames):
    fa = _assign(frames, ["aimd"] * len(frames))
    sq, loss2 = _residual_consistency(fa, w_E=1.0, w_F=0.0)
    assert sq == pytest.approx(loss2, rel=1e-9)


def test_residuals_match_compute_loss_force_block(frames):
    fa = _assign(frames, ["aimd"] * len(frames))
    sq, loss2 = _residual_consistency(fa, w_E=0.0, w_F=1.0)
    assert sq == pytest.approx(loss2, rel=1e-9)


def test_residuals_match_two_groups(frames):
    """Consistency must hold with multiple ref_groups + non-uniform weights too."""
    n = len(frames)
    fa = _assign(frames, ["a"] * (n // 2) + ["b"] * (n - n // 2))
    for fr in fa:
        fr.weight_E = 2.0 if fr.ref_group == "b" else 1.0
    sq, loss2 = _residual_consistency(fa, w_E=1.0, w_F=0.0)
    assert sq == pytest.approx(loss2, rel=1e-9)
