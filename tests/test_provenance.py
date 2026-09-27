"""fit_provenance.toml: the run must be reconstructible from its record —
inputs (dataset hashes), config (controls snapshot), environment, code state
(source hashes), and outcome (fit diagnostics)."""
from __future__ import annotations

import tomllib
from types import SimpleNamespace

from bvff.parsers.controls_parser import Controls
from bvff.parsers.dataset import DatasetEntry
from bvff.core.outputs import save_provenance



def _fake_frame(source: str) -> SimpleNamespace:
    return SimpleNamespace(source=source)


def test_provenance_roundtrip(tmp_path, pto_300k):
    parquet = str(pto_300k)
    controls = Controls(dataset=[DatasetEntry(
        path=parquet, frame_start=100, frame_end=-1, stride=200,
    )])
    train = [_fake_frame(parquet)] * 3
    test  = [_fake_frame(parquet)] * 2
    diag  = {
        "optimizer": "least_squares", "jacobian": "analytic",
        "n_parameters": 27, "n_starts": 1, "log_space": True,
        "lambda_reg": 0.1, "best_loss": 0.42, "initial_loss": 1.0,
        "bound_flags": [{"key": "repulsive.O-O", "value": 1e4,
                         "lower": 0.0, "upper": 1e4,
                         "flag": "at_upper_bound", "space": "log10"}],
    }

    save_provenance(str(tmp_path), controls, train, test, diag,
                    params_file="does_not_exist.toml", elapsed_s=12.34)

    with open(tmp_path / "fit_provenance.toml", "rb") as f:
        doc = tomllib.load(f)

    assert doc["split"] == {"n_train": 3, "n_test": 2,
                            "mode": controls.split_mode, "seed": 0}
    ds = doc["dataset"][0]
    assert ds["path"] == parquet
    assert len(ds["sha256"]) == 16 and ds["sha256"] != "unreadable"
    assert ds["n_train"] == 3 and ds["n_test"] == 2
    # Source hashes present for the core modules; combined hash pins the tree.
    files = doc["source"]["files"]
    assert "bvff/core/potentials.py" in files and "bvff/parsers/defaults.py" in files
    assert len(doc["source"]["combined_sha256"]) == 16
    # Controls snapshot: resolved values, nested sections included.
    assert doc["controls"]["fitting"]["jac"] == "analytic"
    assert doc["controls"]["potentials"]["bv_form"] == "exp"
    assert doc["controls"]["extensions"]["ewald_epsilon"] == float("inf")
    # Fit outcome carried through, including bound flags.
    assert doc["fit"]["best_loss"] == 0.42
    assert doc["fit"]["bound_flags"][0]["flag"] == "at_upper_bound"
    # Environment + elapsed recorded.
    assert "numpy" in doc["environment"]
    assert doc["generated"]["elapsed_s"] == 12.3   # rounded to 0.1 s
    # Missing seed file → no hash key, no crash.
    assert "sha256" not in doc["parameters_seed"]
