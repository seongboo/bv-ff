"""
Cell-adaptive Ewald parameter selection (``ewald_parameters``).

The Ewald total energy is independent of the splitting parameter ``alpha`` once
both the real- and reciprocal-space sums are converged. These tests check that
the auto-selected ``(alpha, kmax)`` give a converged energy (stable vs a larger
kmax) and that it agrees with an independently-chosen larger-alpha setup —
guarding against a regression to the old hardcoded ``alpha=0.3, kmax=5`` that
was under-converged for this cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset        import load_dataset, DatasetEntry
from src.extensions.ewald   import (
    Ewald, ewald_parameters, ewald_alpha, ewald_kmax, clear_kvec_cache,
)


_Q = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}


@pytest.fixture(scope="module")
def frame():
    data = load_dataset(entries=[DatasetEntry(
        path="examples/PbTiO3/pbtio3_222_300K.parquet",
        frame_start=100, frame_end=101, stride=1,
    )])
    return data.frames[0]


def test_adaptive_energy_is_converged(frame):
    alpha, kmax = ewald_parameters(frame.lattice, 6.0)
    clear_kvec_cache(); e   = Ewald(_Q, alpha, kmax,     6.0).energy(frame.lattice, frame.species, frame.positions)
    clear_kvec_cache(); e_hi = Ewald(_Q, alpha, kmax + 3, 6.0).energy(frame.lattice, frame.species, frame.positions)
    assert e == pytest.approx(e_hi, abs=1e-6)


def test_energy_is_alpha_independent_when_converged(frame):
    """Two converged setups with different alpha must agree on the total."""
    a1, k1 = ewald_parameters(frame.lattice, 6.0, accuracy=1e-6)
    a2, k2 = ewald_parameters(frame.lattice, 6.0, accuracy=1e-9)
    clear_kvec_cache(); e1 = Ewald(_Q, a1, k1, 6.0).energy(frame.lattice, frame.species, frame.positions)
    clear_kvec_cache(); e2 = Ewald(_Q, a2, k2, 6.0).energy(frame.lattice, frame.species, frame.positions)
    assert a1 != a2
    assert e1 == pytest.approx(e2, abs=1e-5)


def test_kmax_is_clamped():
    """A pathological tiny cell must not explode the k-grid."""
    _, kmax = ewald_parameters(np.eye(3) * 0.5, 6.0)
    assert 1 <= kmax <= 12


def test_adaptive_kmax_matches_explicit(frame):
    """kmax=None (per-lattice adaptive) must reproduce the explicitly chosen
    converged kmax for the same lattice."""
    alpha, kmax = ewald_parameters(frame.lattice, 6.0)
    clear_kvec_cache(); e_auto = Ewald(_Q, alpha, None, 6.0).energy(frame.lattice, frame.species, frame.positions)
    clear_kvec_cache(); e_expl = Ewald(_Q, alpha, kmax, 6.0).energy(frame.lattice, frame.species, frame.positions)
    assert e_auto == pytest.approx(e_expl, abs=1e-12)


def test_adaptive_kmax_grows_with_cell(frame):
    """A larger cell has shorter reciprocal vectors and needs a larger kmax for
    the same accuracy — the reason kmax must be chosen per lattice for datasets
    mixing cell sizes (primitive DFT anchor + AIMD supercell)."""
    alpha = ewald_alpha(6.0)
    k_small = ewald_kmax(frame.lattice,       alpha)
    k_big   = ewald_kmax(2.0 * frame.lattice, alpha)
    assert k_big > k_small

    # With kmax=None one Ewald instance serves both cells converged: on the
    # big cell it must agree with the explicit converged choice for that cell,
    # not with the small cell's (under-converged) kmax.
    ew_auto = Ewald(_Q, alpha, None, 6.0)
    clear_kvec_cache(); e_auto = ew_auto.energy(2.0 * frame.lattice, frame.species, frame.positions)
    clear_kvec_cache(); e_expl = Ewald(_Q, alpha, k_big, 6.0).energy(2.0 * frame.lattice, frame.species, frame.positions)
    assert e_auto == pytest.approx(e_expl, abs=1e-12)
