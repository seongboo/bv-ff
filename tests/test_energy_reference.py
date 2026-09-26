"""
Energy reference (per-species μ_s) profiled out of the energy loss.

The DFT energy zero is arbitrary, so the fit must depend only on energy
differences. These tests check that:
  - the loss is invariant to adding any Σ_s n_s c_s to the reference energies
  - data generated from the model plus a known reference gives zero energy error
    and the reference is recovered (as E_ref for one composition; as individual
    μ_s when the compositions span the species space)
  - BVFF.energy_ref shifts energies only (forces unchanged)
  - energy_ref survives a save/parse round trip
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from parsers.dataset           import load_dataset, DatasetEntry
from parsers.parameters_parser import parse_parameters, save_parameters
from src.potentials            import BVFF, Repulsive, BV, clear_neighbor_cache
from src.fitting               import compute_loss, fit_energy_reference


_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def frames():
    clear_neighbor_cache()
    return load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=160, stride=10,
    )]).frames


@pytest.fixture(scope="module")
def model():
    """Short-range-only model (no Ewald) so that non-stoichiometric frames
    in the mixed-composition test are allowed."""
    return BVFF([
        Repulsive({"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, cutoff=6.0),
        BV({"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}},
           {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}, cutoff=6.0),
    ])


def _synthetic(frames, model, mu):
    """Frames whose reference energy is exactly E_model + Σ n_s μ_s."""
    out = []
    for fr in frames:
        e = model.energy(fr.lattice, fr.species, fr.positions)
        out.append(replace(fr, energy=e + sum(mu[s] for s in fr.species)))
    return out


def test_loss_invariant_to_reference_shift(frames, model):
    shift = {"Pb": -3.7, "Ti": 12.1, "O": 0.42}
    shifted = [replace(fr, energy=fr.energy + sum(shift[s] for s in fr.species)) for fr in frames]
    l0 = compute_loss(model, frames,  1.0, 0.0, 0.0, False)
    l1 = compute_loss(model, shifted, 1.0, 0.0, 0.0, False)
    assert l1 == pytest.approx(l0, rel=1e-10, abs=1e-12)


def test_zero_energy_error_and_reference_recovered(frames, model):
    mu_true = {"Pb": -5.1, "Ti": -8.3, "O": -7.2}
    syn = _synthetic(frames, model, mu_true)
    assert compute_loss(model, syn, 1.0, 0.0, 0.0, False) < 1e-10
    mu = fit_energy_reference(model, syn)
    # one composition → only E_ref = Σ n_s μ_s is identifiable
    sp = frames[0].species
    assert sum(mu[s] for s in sp) == pytest.approx(sum(mu_true[s] for s in sp), rel=1e-12)


def test_individual_mu_recovered_for_mixed_compositions(frames, model):
    """With compositions spanning all species, each μ_s is identified."""
    fr = frames[0]
    keep_all  = np.arange(len(fr.species))
    drop_O    = keep_all[keep_all != 39]        # (8, 8, 23)
    drop_Pb   = keep_all[keep_all != 0]         # (7, 8, 24)
    variants = [
        replace(fr, species=[fr.species[i] for i in idx], positions=fr.positions[idx],
                forces=fr.forces[idx])
        for idx in (keep_all, drop_O, drop_Pb)
    ]
    clear_neighbor_cache()
    mu_true = {"Pb": -5.1, "Ti": -8.3, "O": -7.2}
    syn = _synthetic(variants, model, mu_true)
    mu = fit_energy_reference(model, syn)
    for s in mu_true:
        assert mu[s] == pytest.approx(mu_true[s], abs=1e-9)


def test_bvff_energy_ref_shifts_energy_only(frames, model):
    fr  = frames[0]
    ref = {"Pb": -5.0, "Ti": -8.0, "O": -7.0}
    shifted = BVFF(model.terms, energy_ref=ref)
    e0, f0 = model.energy_and_forces(fr.lattice, fr.species, fr.positions)
    e1, f1 = shifted.energy_and_forces(fr.lattice, fr.species, fr.positions)
    assert e1 - e0 == pytest.approx(sum(ref[s] for s in fr.species), rel=1e-12)
    assert shifted.energy(fr.lattice, fr.species, fr.positions) == pytest.approx(e1, rel=1e-12)
    np.testing.assert_array_equal(f0, f1)


def test_energy_ref_save_parse_roundtrip(tmp_path):
    p = parse_parameters(str(_ROOT / "parameters.toml"))
    p.energy_ref = {"O": -7.25, "Pb": -3.5, "Ti": -9.125}
    path = tmp_path / "params.toml"
    save_parameters(p, str(path))
    q = parse_parameters(str(path))
    assert q.energy_ref == p.energy_ref
