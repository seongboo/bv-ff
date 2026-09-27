"""
Train/test split modes (src/main.py split_frames) and the dataset ``split``
value normalization (parsers/controls_parser._norm_split).

The random within-trajectory split of time-correlated AIMD frames guarantees
train≈test RMSE by construction; ``split_mode="block"`` (hold out the
temporally last fraction of each trajectory) and ``split="test"`` (hold out a
whole file/condition) are the honest generalization protocols. These tests pin
the exact routing semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset         import Dataset, Frame
from parsers.controls_parser import _norm_split
from src.main                import split_frames


def _mk(idx: int, source: str, split=True, ref_group="default") -> Frame:
    return Frame(
        index=idx, lattice=np.eye(3) * 4.0, species=["Ti"],
        positions=np.zeros((1, 3)), energy=0.0, forces=np.zeros((1, 3)),
        stress=None, source=source, ref_group=ref_group, split=split,
    )


def _data(frames) -> Dataset:
    return Dataset(frames=frames, n_atoms=1, species=["Ti"])


def test_block_split_holds_out_temporal_tail():
    frames = [_mk(i, "a.pq") for i in range(10)]
    tr, te = split_frames(_data(frames), 0.8, seed=0, mode="block")
    assert [f.index for f in tr] == list(range(8))
    assert [f.index for f in te] == [8, 9]


def test_block_split_is_per_source():
    frames = [_mk(i, "a") for i in range(10)] + [_mk(100 + i, "b") for i in range(5)]
    tr, te = split_frames(_data(frames), 0.8, seed=0, mode="block")
    assert sorted(f.index for f in te) == [8, 9, 104]


def test_test_pinned_entry_goes_entirely_to_test():
    frames = ([_mk(i, "a") for i in range(5)]
              + [_mk(100 + i, "b", split="test") for i in range(3)])
    tr, te = split_frames(_data(frames), 0.8, seed=0, mode="random")
    assert all(f.source != "b" for f in tr)
    assert sum(f.source == "b" for f in te) == 3


def test_train_pinned_anchor_goes_entirely_to_train():
    frames = [_mk(i, "a") for i in range(5)] + [_mk(100, "anchor", split=False)]
    tr, te = split_frames(_data(frames), 0.8, seed=0, mode="block")
    assert any(f.source == "anchor" for f in tr)
    assert all(f.source != "anchor" for f in te)


def test_random_split_reproducible_and_mode_validated():
    frames = [_mk(i, "a") for i in range(20)]
    tr1, _ = split_frames(_data(frames), 0.8, seed=3, mode="random")
    tr2, _ = split_frames(_data(frames), 0.8, seed=3, mode="random")
    assert [f.index for f in tr1] == [f.index for f in tr2]
    with pytest.raises(ValueError, match="split mode"):
        split_frames(_data(frames), 0.8, seed=3, mode="shuffle")


def test_test_only_ref_group_raises():
    """A split="test" holdout whose ref_group has no training frames has an
    undefined energy offset — must error, not silently distort the test RMSE."""
    frames = ([_mk(i, "a", ref_group="g1") for i in range(5)]
              + [_mk(100 + i, "b", split="test", ref_group="g2") for i in range(2)])
    with pytest.raises(ValueError, match="no training frames"):
        split_frames(_data(frames), 0.8, seed=0, mode="random")


def test_norm_split_values():
    assert _norm_split(True, "x") is True
    assert _norm_split(False, "x") is False
    assert _norm_split(1, "x") is True        # 1/0 style, like the use_* toggles
    assert _norm_split(0, "x") is False
    assert _norm_split("train", "x") is False
    assert _norm_split("TEST", "x") == "test"
    with pytest.raises(ValueError, match="split must be"):
        _norm_split("holdout", "x")
    with pytest.raises(ValueError, match="split must be"):
        _norm_split(2, "x")


def test_gen_controls_emits_refit_keys(tmp_path):
    """The recommended refit recipe (Buckingham repulsion, exp BV form, frozen
    charges, Tikhonov, block split) must be generatable — not hand-edited —
    and the result must parse."""
    import tomllib
    from scripts.gen_controls import _build_parser, build_toml

    args = _build_parser().parse_args([
        "data.parquet", "--buckingham", "--bv-form", "exp",
        "--split-mode", "block", "--lambda-reg", "1e-3", "--use-stress",
    ])
    if args.buckingham and args.repulsive:      # mirrors main()'s exclusivity rule
        args.repulsive = False
    cfg = tomllib.loads(build_toml(args))

    assert cfg["split_mode"] == "block"
    assert cfg["potentials"]["use_buckingham"] == 1
    assert cfg["potentials"]["use_repulsive"] == 0
    assert cfg["potentials"]["bv_form"] == "exp"
    assert cfg["fitting"]["lambda_reg"] == pytest.approx(1e-3)
    assert cfg["fitting"]["use_stress"] == 1
    assert cfg["fitting"]["fit_charges"] == 0
