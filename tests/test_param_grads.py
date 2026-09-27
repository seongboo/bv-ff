"""
Analytic parameter gradients (Phase 2) vs central finite differences.

For every term and every fitted parameter θ, ``param_grads`` must return
(∂E/∂θ, ∂F/∂θ) matching an independent FD of energy() and forces() under a
θ-perturbation — with the cutoff taper both off and on (the taper is
θ-independent but enters every chain rule). Run on the real 40-atom PbTiO3
frame so all scatter/gather paths are exercised.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials import (
    Repulsive, Buckingham, BV, BVV, Angle, BVFF,
    clear_neighbor_cache,
)


CUTOFF = 6.0

_BV_SP  = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.7},
           "O":  {"V0": 2.0, "S": 0.4}}
_BV_PP  = {"O-Pb": {"r0": 2.06, "C": 6.0, "b": 0.40},
           "O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.37}}
_BVV_SP = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.2},
           "O":  {"W0": 0.2, "D": 0.05}}
_BUCK   = {"O-Ti": {"A": 700.0, "rho": 0.35, "C": 30.0},
           "O-Pb": {"A": 900.0, "rho": 0.38, "C": 20.0}}
_REP    = {"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = "examples/PbTiO3/pbtio3_222_300K.parquet",
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    return data.frames[0]


def _perturbed(builder, key: str, val: float):
    """Build the term with parameter ``key`` set to ``val`` (deep-copied dicts)."""
    return builder(key, val)


def _fd_check(pot, builder, key: str, theta: float, frame,
              rel: float = 3e-6, h_rel: float = 1e-6):
    """Central-difference ∂E/∂θ and ∂F/∂θ vs the analytic column."""
    grads = pot.param_grads(frame.lattice, frame.species, frame.positions)
    assert key in grads, f"{type(pot).__name__} missing key {key}"
    dE_an, dF_an = grads[key]

    h  = h_rel * max(1.0, abs(theta))
    pp = _perturbed(builder, key, theta + h)
    pm = _perturbed(builder, key, theta - h)
    e_p = pp.energy(frame.lattice, frame.species, frame.positions)
    e_m = pm.energy(frame.lattice, frame.species, frame.positions)
    f_p = pp.forces(frame.lattice, frame.species, frame.positions)
    f_m = pm.forces(frame.lattice, frame.species, frame.positions)

    dE_fd = (e_p - e_m) / (2 * h)
    dF_fd = (f_p - f_m) / (2 * h)
    scale_E = max(1.0, abs(dE_fd))
    scale_F = max(1.0, np.abs(dF_fd).max())
    assert dE_an == pytest.approx(dE_fd, abs=rel * scale_E), key
    assert np.allclose(dF_an, dF_fd, atol=rel * scale_F), key


# ──────────────────────────────────────────────
# Per-term FD sweeps (taper off and on)
# ──────────────────────────────────────────────

@pytest.mark.parametrize("sw", [0.0, 1.0])
def test_repulsive_param_grads(frame, sw):
    def build(key, val):
        B = dict(_REP)
        B[key.split(".")[1]] = val
        return Repulsive(B=B, cutoff=CUTOFF, smooth_width=sw)
    pot = Repulsive(B=_REP, cutoff=CUTOFF, smooth_width=sw)
    for pair, val in _REP.items():
        _fd_check(pot, build, f"repulsive.{pair}", val, frame)


@pytest.mark.parametrize("sw", [0.0, 1.0])
def test_buckingham_param_grads(frame, sw):
    def build(key, val):
        _, pair, attr = key.split(".")
        params = copy.deepcopy(_BUCK)
        params[pair][attr] = val
        return Buckingham(params=params, cutoff=CUTOFF, smooth_width=sw)
    pot = Buckingham(params=_BUCK, cutoff=CUTOFF, smooth_width=sw)
    for pair, p in _BUCK.items():
        for attr in ("A", "rho", "C"):
            _fd_check(pot, build, f"buckingham.{pair}.{attr}", p[attr], frame)


@pytest.mark.parametrize("sw", [0.0, 1.0])
@pytest.mark.parametrize("form", ["exp", "power"])
def test_bv_param_grads(frame, form, sw):
    def build(key, val):
        sp, pp = copy.deepcopy(_BV_SP), copy.deepcopy(_BV_PP)
        parts = key.split(".")
        if parts[1] == "species":
            sp[parts[2]][parts[3]] = val
        else:
            pp[parts[2]][parts[3]] = val
        return BV(species_params=sp, pair_params=pp, cutoff=CUTOFF,
                  form=form, smooth_width=sw)
    pot = BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
             form=form, smooth_width=sw)
    for atom, sp in _BV_SP.items():
        _fd_check(pot, build, f"BV.species.{atom}.V0", sp["V0"], frame)
        _fd_check(pot, build, f"BV.species.{atom}.S",  sp["S"],  frame)
    shape = "b" if form == "exp" else "C"
    for pair, p in _BV_PP.items():
        _fd_check(pot, build, f"BV.pairs.{pair}.r0", p["r0"], frame)
        _fd_check(pot, build, f"BV.pairs.{pair}.{shape}", p[shape], frame)


@pytest.mark.parametrize("sw", [0.0, 1.0])
@pytest.mark.parametrize("form", ["exp", "power"])
def test_bvv_param_grads(frame, form, sw):
    def build(key, val):
        sp, pp = copy.deepcopy(_BVV_SP), copy.deepcopy(_BV_PP)
        parts = key.split(".")
        if parts[1] == "pairs":
            pp[parts[2]][parts[3]] = val
        else:
            sp[parts[1]][parts[2]] = val
        return BVV(species_params=sp, pair_params=pp, cutoff=CUTOFF,
                   form=form, smooth_width=sw)
    pot = BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF,
              form=form, smooth_width=sw)
    for atom, sp in _BVV_SP.items():
        _fd_check(pot, build, f"BVV.{atom}.W0", sp["W0"], frame)
        _fd_check(pot, build, f"BVV.{atom}.D",  sp["D"],  frame)
    shape = "b" if form == "exp" else "C"
    for pair, p in _BV_PP.items():
        _fd_check(pot, build, f"BVV.pairs.{pair}.r0", p["r0"], frame)
        _fd_check(pot, build, f"BVV.pairs.{pair}.{shape}", p[shape], frame)


@pytest.mark.parametrize("k", [0.0, 0.001])
def test_angle_param_grads(frame, k):
    """Linear in k — and the k=0 seed must still get a non-trivial gradient
    (the k=1 clone path)."""
    def build(key, val):
        return Angle(k=val, cutoff=3.5, smooth_width=0.5)
    pot = Angle(k=k, cutoff=3.5, smooth_width=0.5)
    _fd_check(pot, build, "angle.k", max(k, 0.001), frame)
    dE, dF = pot.param_grads(frame.lattice, frame.species, frame.positions)["angle.k"]
    assert dE > 0.0 and np.abs(dF).max() > 0.0


def test_bvff_merges_all_terms(frame):
    bvff = BVFF([
        Repulsive(B=_REP, cutoff=CUTOFF),
        BV(species_params=_BV_SP, pair_params=_BV_PP, cutoff=CUTOFF, form="exp"),
        BVV(species_params=_BVV_SP, pair_params=_BV_PP, cutoff=CUTOFF, form="exp"),
    ])
    grads = bvff.param_grads(frame.lattice, frame.species, frame.positions)
    expected = (
        {f"repulsive.{p}" for p in _REP}
        | {f"BV.species.{a}.{t}" for a in _BV_SP for t in ("V0", "S")}
        | {f"BV.pairs.{p}.{t}" for p in _BV_PP for t in ("r0", "b")}
        | {f"BVV.{a}.{t}" for a in _BVV_SP for t in ("W0", "D")}
        | {f"BVV.pairs.{p}.{t}" for p in _BV_PP for t in ("r0", "b")}
    )
    assert set(grads) == expected
