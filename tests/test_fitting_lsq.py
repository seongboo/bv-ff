"""
Least-squares fitting: residual definition and end-to-end parameter recovery.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from parsers.dataset           import load_dataset, DatasetEntry
from parsers.controls_parser   import parse_controls
from parsers.parameters_parser import parse_parameters
from src.main                  import build_bvff
from src.potentials            import clear_neighbor_cache
from src.fitting import (
    fit, compute_loss, residual_vector, data_scales,
    params_to_vector, vector_to_params, build_bounds, free_mask, frames_composition,
)


_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def setup():
    clear_neighbor_cache()
    c = parse_controls(str(_ROOT / "controls.toml"))
    p = parse_parameters(str(_ROOT / "parameters.toml"))
    frames = load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=-1, stride=200,
    )]).frames                                            # 5 frames
    build = lambda q: build_bvff(c, q)
    bt = build(p)
    mu = {"Pb": -3.0, "Ti": -5.0, "O": -8.0}
    syn = []
    for fr in frames:
        e, f = bt.energy_and_forces(fr.lattice, fr.species, fr.positions)
        syn.append(replace(fr, energy=e + sum(mu[s] for s in fr.species), forces=f))
    return c, p, build, frames, syn


def test_residual_norm_matches_normalised_mse(setup):
    """||r||² = w_E·MSE_E/σ_E² + w_F·MSE_F/σ_F², with RMSE_k from compute_loss."""
    c, p, build, frames, _ = setup
    b  = build(p)
    sc = data_scales(frames)
    wE, wF = 0.7, 1.3
    r  = residual_vector(b, frames, wE, wF, 1.0, sc)
    rmse_E = compute_loss(b, frames, 1, 0, 0, False)
    rmse_F = compute_loss(b, frames, 0, 1, 0, False)
    assert r @ r == pytest.approx(wE * rmse_E**2 / sc["E"]**2 + wF * rmse_F**2 / sc["F"]**2, rel=1e-10)
    # compute_loss with scales is the σ-normalised sum of RMSEs
    assert compute_loss(b, frames, wE, wF, 0, False, scales=sc) == pytest.approx(
        wE * rmse_E / sc["E"] + wF * rmse_F / sc["F"], rel=1e-12)


def test_lsq_recovers_synthetic_parameters(setup):
    """Noise-free data from known parameters; start from a ±20% perturbation
    of the free parameters. With V0 held fixed the fit must recover them."""
    c, p_true, build, _, syn = setup
    comp = frames_composition(syn)
    xa, ka = params_to_vector(p_true, c.potentials, comp)
    m = free_mask(ka, c.fitting.fixed)
    keys = [k for k, b in zip(ka, m) if b]
    lo, hi = np.array(build_bounds(keys)).T
    x0 = np.clip(xa[m] * np.random.default_rng(4).uniform(0.8, 1.2, m.sum()), lo, hi)
    p0 = vector_to_params(x0, keys, p_true, comp)

    p_fit, loss = fit(build, p0, c.potentials, syn, 1.0, 1.0, 1.0, False,
                      fixed=c.fitting.fixed, optimizer="lsq", n_starts=2, seed=0)
    x_fit, _ = params_to_vector(p_fit, c.potentials, comp)
    x_true = xa
    rel = np.abs(x_fit - x_true) / np.where(x_true != 0, np.abs(x_true), 1.0)
    assert loss < 1e-6          # Σ RMSE/σ (not squared); lsq stops on ftol=1e-10 in ½‖r‖²
    # production tolerances (ftol = xtol = 1e-10) give ~1e-6 relative accuracy;
    # with 1e-14 the same recovery reaches ~1e-13 (see commit message).
    assert rel.max() < 1e-5, dict(zip(ka, rel))
    E_ref = sum(p_fit.energy_ref[s] for s in syn[0].species)
    # absolute cell energies are O(10^2–10^3) eV, so ~1e-6 parameter accuracy
    # propagates to ~1e-5 eV per 40-atom cell in the fitted reference
    assert E_ref == pytest.approx(sum({"Pb": -3.0, "Ti": -5.0, "O": -8.0}[s] for s in syn[0].species), abs=1e-4)
