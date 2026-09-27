"""
Phase 2 fitting tests: the assembled analytic residual Jacobian vs finite
differences, the FD fallback for charge columns, and multi-start
least_squares.

The Jacobian test exercises every assembly wrinkle at once: log-space chain
rule, per-ref_group weighted energy centering, per-frame force-row scaling,
and the Tikhonov block.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.controls_parser import PotentialControls
from bvff.parsers.dataset import load_dataset, DatasetEntry
from bvff.parsers.parameters_parser import (
    Parameters, RepulsiveParams, BuckinghamParams, BuckinghamPair,
    BVParams, BVSpecies, BVPair, BVVParams, BVVSpecies, CoulombParams,
)
from bvff.core.fitting import (
    params_to_vector, build_bounds, ParamTransform, build_frame_weights,
    reference_scales, _residuals_z, _jacobian_z, fit,
)
from bvff.core.potentials import (
    Coulomb, Repulsive, Buckingham, BV, BVV, BVFF, clear_neighbor_cache,
)


@pytest.fixture(scope="module")
def frames(pto_300k):
    clear_neighbor_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = str(pto_300k),
        frame_start = 100, frame_end = 181, stride = 20,
    )])
    frs = data.frames
    assert len(frs) >= 4
    # Two ref_groups → exercises the per-group energy centering in J.
    for fr in frs[2:]:
        fr.ref_group = "g2"
    return frs[:4]


def _make_params() -> Parameters:
    p = Parameters()
    p.cutoff       = 6.0
    p.smooth_width = 1.0
    p.repulsive    = RepulsiveParams(B={"O-O": 1.83, "O-Ti": 1.28})
    p.buckingham   = BuckinghamParams(
        pairs={"O-Ti": BuckinghamPair(A=700.0, rho=0.35, C=30.0)})
    p.BV = BVParams(
        species={"Pb": BVSpecies(2.0, 0.5), "Ti": BVSpecies(4.0, 0.7),
                 "O": BVSpecies(2.0, 0.4)},
        pairs={"O-Pb": BVPair(2.06, 6.0, 0.40), "O-Ti": BVPair(1.81, 5.2, 0.37)},
    )
    p.BVV = BVVParams(
        species={"Pb": BVVSpecies(0.5, 0.1), "Ti": BVVSpecies(0.3, 0.2),
                 "O": BVVSpecies(0.2, 0.05)},
        pairs={"O-Pb": BVPair(2.06, 6.0, 0.40), "O-Ti": BVPair(1.81, 5.2, 0.37)},
    )
    return p


def _builder(p: Parameters) -> BVFF:
    terms = [
        Repulsive(B=p.repulsive.B, cutoff=p.cutoff, smooth_width=p.smooth_width),
        Buckingham(
            params={k: {"A": v.A, "rho": v.rho, "C": v.C}
                    for k, v in p.buckingham.pairs.items()},
            cutoff=p.cutoff, smooth_width=p.smooth_width),
        BV(species_params={a: {"V0": s.V0, "S": s.S}
                           for a, s in p.BV.species.items()},
           pair_params={k: {"r0": q.r0, "C": q.C, "b": q.b}
                        for k, q in p.BV.pairs.items()},
           cutoff=p.cutoff, form="exp", smooth_width=p.smooth_width),
        BVV(species_params={a: {"W0": s.W0, "D": s.D}
                            for a, s in p.BVV.species.items()},
            pair_params={k: {"r0": q.r0, "C": q.C, "b": q.b}
                         for k, q in p.BVV.pairs.items()},
            cutoff=p.cutoff, form="exp", smooth_width=p.smooth_width),
    ]
    if p.coulomb.charges:
        terms.insert(0, Coulomb(charges=p.coulomb.charges, cutoff=p.cutoff,
                                smooth_width=p.smooth_width))
    return BVFF(terms)


def _pc(coulomb: bool = False) -> PotentialControls:
    return PotentialControls(
        use_coulomb=coulomb, use_repulsive=True, use_buckingham=True,
        use_BV=True, use_BVV=True, use_angle=False, bv_form="exp",
    )


def _setup(frames, params, pc, fit_charges=False, lambda_reg=0.0):
    x0, keys  = params_to_vector(params, pc, None, fit_charges)
    bounds    = build_bounds(keys)
    transform = ParamTransform(keys, bounds, log_space=True)
    lb = np.array([b[0] for b in bounds]); ub = np.array([b[1] for b in bounds])
    x0_seed   = transform.from_opt(transform.to_opt(np.clip(x0, lb, ub)))
    gid, wE, wF, _, _ = build_frame_weights(frames, "count")
    norm_E, norm_F, _ = reference_scales(frames, wE, gid)
    counts = {}
    res_args = (keys, params, None, counts, _builder, frames,
                1.0, 1.0, norm_E, norm_F, 1, wE, wF, gid, lambda_reg, x0_seed)
    return transform, res_args, transform.to_opt(x0_seed), keys


def _fd_jacobian(z, transform, res_args, h_rel=1e-7):
    r0 = _residuals_z(z, transform, *res_args)
    J  = np.zeros((r0.size, z.size))
    for k in range(z.size):
        h  = h_rel * max(1.0, abs(z[k]))
        zp = z.copy(); zp[k] += h
        zm = z.copy(); zm[k] -= h
        J[:, k] = (_residuals_z(zp, transform, *res_args)
                   - _residuals_z(zm, transform, *res_args)) / (2 * h)
    return J


def test_jacobian_matches_fd(frames):
    """Full assembled J vs central differences of the actual residual —
    log-space chain, group centering, force scaling, Tikhonov rows."""
    transform, res_args, z, keys = _setup(
        frames, _make_params(), _pc(), lambda_reg=0.05)
    J_an = _jacobian_z(z, transform, *res_args)
    J_fd = _fd_jacobian(z, transform, res_args)
    assert J_an.shape == J_fd.shape
    scale = np.abs(J_fd).max()
    assert np.allclose(J_an, J_fd, atol=3e-5 * max(1.0, scale)), \
        f"max dev {np.abs(J_an - J_fd).max():.3e} vs scale {scale:.3e}"


def test_jacobian_charge_fd_fallback(frames):
    """With fit_charges=1 the coulomb.* columns have no analytic coverage and
    must come from the FD fallback — still matching a full FD Jacobian."""
    params = _make_params()
    params.coulomb = CoulombParams(
        charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8})
    transform, res_args, z, keys = _setup(
        frames[:2], params, _pc(coulomb=True), fit_charges=True)
    assert any(k.startswith("coulomb.") for k in keys)
    J_an = _jacobian_z(z, transform, *res_args)
    J_fd = _fd_jacobian(z, transform, res_args)
    scale = np.abs(J_fd).max()
    # Charge columns are one-sided FD (h=1e-6), so give them a looser band.
    assert np.allclose(J_an, J_fd, atol=2e-4 * max(1.0, scale))


def test_multistart_least_squares_runs(frames):
    """n_starts>1 on the LS path: runs, logs per-start results, and the kept
    result is no worse than the pure seed start."""
    params = _make_params()
    fitted, loss = fit(
        bvff_builder=_builder, params=params, controls_potentials=_pc(),
        train_frames=frames, w_E=1.0, w_F=1.0, w_S=1.0, has_stress=False,
        maxiter=8, n_starts=3, n_jobs=1, optimizer="least_squares",
        jac="analytic", fit_charges=False,
    )
    fitted1, loss1 = fit(
        bvff_builder=_builder, params=_make_params(), controls_potentials=_pc(),
        train_frames=frames, w_E=1.0, w_F=1.0, w_S=1.0, has_stress=False,
        maxiter=8, n_starts=1, n_jobs=1, optimizer="least_squares",
        jac="analytic", fit_charges=False,
    )
    assert np.isfinite(loss) and np.isfinite(loss1)
    assert loss <= loss1 + 1e-9


def test_controls_jac_validation(tmp_path):
    from bvff.parsers.controls_parser import parse_controls
    (tmp_path / "d.extxyz").write_text("")
    bad = tmp_path / "controls.toml"
    bad.write_text(
        f'dataset = "{tmp_path}/d.extxyz"\n[fitting]\njac = "magic"\n'
    )
    with pytest.raises(ValueError, match="fitting.jac"):
        parse_controls(str(bad))
