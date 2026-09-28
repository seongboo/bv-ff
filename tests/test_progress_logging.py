"""
Progress output of long-running stages (PLAN 1-1): the optimizer and MD loops
must emit periodic log lines instead of staying silent until they finish.
The time throttle is patched to 0 s so every iteration is eligible to log.
"""
import logging

import pytest

import bvff.core.fitting as fitting
import bvff.core.outputs as outputs
from bvff.core.outputs import LogThrottle
from bvff.core.potentials import clear_neighbor_cache
from bvff.parsers.dataset import load_dataset, DatasetEntry

from test_fitting import _params, _builder, _PC


@pytest.fixture(scope="module")
def frames(pto_300k):
    clear_neighbor_cache()
    return load_dataset(entries=[DatasetEntry(
        path=str(pto_300k), frame_start=100, frame_end=120, stride=10,
    )]).frames


@pytest.fixture
def no_throttle(monkeypatch):
    monkeypatch.setattr(fitting, "LogThrottle", lambda: LogThrottle(0.0))


def _fit_lsq(frames, **kw):
    return fitting.fit(
        _builder, _params(), _PC, frames,
        w_E=1.0, w_F=1.0, w_S=1.0, has_stress=False, use_stress=False,
        optimizer="least_squares", **kw,
    )


def test_log_throttle(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(outputs.time, "monotonic", lambda: now[0])
    th = LogThrottle(10.0)
    assert th.ready()                   # first call always passes
    now[0] += 5.0
    assert not th.ready()               # within the interval
    assert th.ready(force=True)         # forced, restarts the interval
    now[0] += 9.0
    assert not th.ready()
    now[0] += 1.0
    assert th.ready()                   # interval elapsed


def test_least_squares_logs_iterations(frames, caplog, no_throttle):
    with caplog.at_level(logging.INFO, logger="bvff"):
        _fit_lsq(frames, maxiter=4, n_jobs=1)
    iters = [r.getMessage() for r in caplog.records if "LSQ iter" in r.getMessage()]
    assert len(iters) >= 2
    assert "residual evals" in iters[0] and "elapsed" in iters[0]


def test_multistart_logs_each_start_as_it_finishes(frames, caplog):
    with caplog.at_level(logging.INFO, logger="bvff"):
        _fit_lsq(frames, maxiter=3, n_jobs=2, n_starts=2)
    done = [r.getMessage() for r in caplog.records if " done (" in r.getMessage()]
    assert len(done) == 2
    assert {"(1/2", "(2/2"} == {m.split(" done ")[1][:4] for m in done}
    assert sorted(m.split()[1] for m in done) == ["0", "1"]   # both starts reported


def test_tc_scan_logs_md_progress(caplog, monkeypatch):
    from bvff.core.calculator import BVFFCalculator
    from bvff.tools.ferroelectric import ideal_perovskite
    from bvff.tools.tc_scan import scan_temperatures
    from test_phase3_fe import _toy_bvff

    # tc_scan imports LogThrottle from bvff.core.outputs at call time
    monkeypatch.setattr(outputs, "LogThrottle", lambda: LogThrottle(0.0))
    log = logging.getLogger("bvff.test.tc")
    with caplog.at_level(logging.INFO, logger="bvff.test.tc"):
        scan_temperatures(ideal_perovskite(a=3.97, rep=(2, 2, 2), A="Pb"),
                          BVFFCalculator(_toy_bvff()), temps=[50.0],
                          steps=6, equil=2, sample_every=2, timestep_fs=1.0,
                          logger=log)
    steps = [r.getMessage() for r in caplog.records if "| step " in r.getMessage()]
    assert len(steps) == 6
    assert "step      6/6" in steps[-1]
    assert all("ETA" in m for m in steps)
