"""
Regression tests for the Phase-1 confirmed-bug fixes outside potentials.py:

- auto-generated BV pairs must be cation-anion only (no O-O anion-anion pair)
- PotentialControls dataclass default bv_form must match DEFAULT_CONTROLS
  (they drifted once: "power" vs "exp")
- smooth_width: parsed, saved, validated, defaulted
- save_results must raise on an unknown task instead of silently saving nothing
- mixed-cell (ragged) force arrays must save instead of crashing np.array
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from bvff.parsers.controls_parser import PotentialControls
from bvff.parsers.defaults import DEFAULT_CONTROLS, DEFAULT_PARAMETERS
from bvff.parsers.parameters_parser import (
    Parameters, _generate_bv_pairs, generate_parameters,
    save_parameters, parse_parameters, _validate,
)
from bvff.core.potentials import Repulsive, BVFF
from bvff.core.outputs import save_results, predict_with_progress


# ──────────────────────────────────────────────
# BV pair auto-generation: cation-anion only
# ──────────────────────────────────────────────

def test_bv_pairs_exclude_anion_anion():
    pairs = _generate_bv_pairs(["Pb", "Ti", "O"])
    assert sorted(pairs) == ["O-Pb", "O-Ti"]          # no O-O, no Pb-Ti


def test_bv_pairs_no_anion_fallback():
    # No recognised anion → all pairs, user prunes by hand (unchanged behavior).
    pairs = _generate_bv_pairs(["Pb", "Ti"])
    assert sorted(pairs) == ["Pb-Pb", "Pb-Ti", "Ti-Ti"]


def test_generated_parameters_have_no_oo_bv_pair():
    params = generate_parameters(
        species=["Pb", "Ti", "O"], use_coulomb=True, use_repulsive=True,
        use_BV=True, use_BVV=True, use_angle=False,
    )
    assert "O-O" not in params.BV.pairs
    assert "O-O" not in params.BVV.pairs
    assert "O-O" in params.repulsive.B                # pair terms still get it


# ──────────────────────────────────────────────
# bv_form default drift
# ──────────────────────────────────────────────

def test_bv_form_dataclass_matches_defaults():
    assert PotentialControls().bv_form == DEFAULT_CONTROLS["potentials"]["bv_form"]


# ──────────────────────────────────────────────
# smooth_width plumbing
# ──────────────────────────────────────────────

def test_smooth_width_default():
    assert Parameters().smooth_width == DEFAULT_PARAMETERS["smooth_width"]


def test_smooth_width_roundtrip(tmp_path):
    p = Parameters(cutoff=6.0, smooth_width=0.5)
    path = tmp_path / "params.toml"
    save_parameters(p, str(path))
    reparsed = parse_parameters(str(path), validate=False)
    assert reparsed.smooth_width == 0.5


def test_smooth_width_default_when_absent(tmp_path):
    # Old files (pre-Phase-1) have no smooth_width key.
    path = tmp_path / "params.toml"
    path.write_text("cutoff = 6.0\n")
    reparsed = parse_parameters(str(path), validate=False)
    assert reparsed.smooth_width == DEFAULT_PARAMETERS["smooth_width"]


@pytest.mark.parametrize("bad", [-0.1, 6.0, 7.5])
def test_smooth_width_validation(bad):
    with pytest.raises(ValueError, match="smooth_width"):
        _validate(Parameters(cutoff=6.0, smooth_width=bad))


# ──────────────────────────────────────────────
# save_results / ragged forces
# ──────────────────────────────────────────────

def _quiet_logger():
    logger = logging.getLogger("bvff-test")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _fake_frame(n_atoms: int) -> SimpleNamespace:
    rng = np.random.default_rng(n_atoms)
    return SimpleNamespace(
        lattice   = np.eye(3) * 8.0,
        species   = ["O"] * n_atoms,
        positions = rng.random((n_atoms, 3)),
    )


def _tiny_bvff() -> BVFF:
    return BVFF([Repulsive(B={"O-O": 1.0}, cutoff=4.0)])


def test_save_results_unknown_task_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown task"):
        save_results(_tiny_bvff(), str(tmp_path), "forces",   # typo for "force"
                     train_frames=[], test_frames=[], logger=_quiet_logger())


def test_ragged_forces_stack_and_save(tmp_path):
    """A mixed 5-atom + 8-atom dataset used to crash np.array(forces) with
    'inhomogeneous shape'; now it stacks to an object array and round-trips
    through save_results."""
    frames = [_fake_frame(5), _fake_frame(8)]
    forces = predict_with_progress(_tiny_bvff(), frames, "force", "train",
                                   _quiet_logger())
    assert forces.dtype == object
    assert forces[0].shape == (5, 3) and forces[1].shape == (8, 3)

    save_results(_tiny_bvff(), str(tmp_path), "both",
                 train_frames=frames, test_frames=frames[:1],
                 logger=_quiet_logger())
    loaded = np.load(tmp_path / "train_forces.npy", allow_pickle=True)
    assert loaded[1].shape == (8, 3)
    np.testing.assert_allclose(loaded[0], forces[0])


def test_homogeneous_forces_stay_regular(tmp_path):
    frames = [_fake_frame(6), _fake_frame(6)]
    forces = predict_with_progress(_tiny_bvff(), frames, "force", "train",
                                   _quiet_logger())
    assert forces.dtype != object
    assert forces.shape == (2, 6, 3)
