"""
C² smooth cutoff (quintic switching on [r_c − Δ, r_c]).

Checks: the switching function itself; energy/force of a dimer vanish
continuously at r_c with the expected (r_c − r)³ / (r_c − r)² scaling;
finite-difference forces of every switched term with many pairs inside the
switching shell; and the analytic virial vs FD strain derivative at the
original 6.0 Å cutoff, which failed with a hard cutoff because an image pair
sits 1.2e-5 Å from it (see test_potentials_stress.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials  import (
    switching, Repulsive, Buckingham, BV, BVV, clear_neighbor_cache,
)
from tests.test_potentials_stress import _fd_stress


_ROOT  = Path(__file__).resolve().parent.parent
RC, DW = 6.0, 1.0
_SP  = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_PP  = {"O-Pb": {"r0": 2.06, "C": 6.0, "b": 0.37}, "O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.40}}
_VSP = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.2, "D": 0.05}}
_BUCK = {"O-O":  {"A": 1500.0, "rho": 0.30, "C": 12.0},
         "O-Pb": {"A": 2000.0, "rho": 0.35, "C": 18.0},
         "O-Ti": {"A":  877.0, "rho": 0.38, "C":  9.0}}


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    return load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=101, stride=1,
    )]).frames[0]


def test_switching_function():
    r = np.linspace(RC - DW - 0.5, RC + 0.5, 20001)
    S, dS = switching(r, RC, DW)
    assert np.all(S[r <= RC - DW] == 1.0) and np.all(S[r >= RC] == 0.0)
    assert np.all((S >= 0) & (S <= 1))
    h = r[1] - r[0]
    np.testing.assert_allclose(np.gradient(S, h)[1:-1], dS[1:-1], atol=1e-6)
    d2 = np.gradient(dS, h)
    # S″ → 0 at both ends (C² matching to the constant pieces)
    for r_end in (RC - DW, RC):
        k = np.argmin(np.abs(r - r_end))
        assert abs(d2[k]) < 1e-2
    # width 0 → hard cutoff
    S0, dS0 = switching(r, RC, 0.0)
    assert np.all(S0 == 1.0) and np.all(dS0 == 0.0)


def test_dimer_vanishes_smoothly_at_cutoff():
    """E ∝ (r_c − r)³ and F ∝ (r_c − r)² as r → r_c⁻ (S = 10(1−x)³ + O((1−x)⁴))."""
    rep = Repulsive({"O-Ti": 2.5}, cutoff=RC, cutoff_width=DW)
    L = np.eye(3) * 30.0
    def ef(d):
        x = np.array([[0, 0, 0], [d / 30.0, 0, 0]])
        clear_neighbor_cache()
        return rep.energy(L, ["Ti", "O"], x), rep.forces(L, ["Ti", "O"], x)[1, 0]
    e1, f1 = ef(RC - 1e-3)
    e2, f2 = ef(RC - 2e-3)
    assert e2 / e1 == pytest.approx(8.0, rel=1e-2)
    assert f2 / f1 == pytest.approx(4.0, rel=1e-2)
    e0, f0 = ef(RC + 1e-3)
    assert e0 == 0.0 and f0 == 0.0


def _fd_check(pot, frame, atoms=(0, 8, 16, 30), eps=1e-5, atol=1e-6):
    L, sp, x = frame.lattice, frame.species, frame.positions
    inv = np.linalg.inv(L)
    f = pot.forces(L, sp, x)
    for a in atoms:
        for k in range(3):
            xp = x.copy(); xp[a] += eps * inv[k]
            xm = x.copy(); xm[a] -= eps * inv[k]
            fd = -(pot.energy(L, sp, xp) - pot.energy(L, sp, xm)) / (2 * eps)
            assert fd == pytest.approx(f[a, k], abs=atol), (type(pot).__name__, a, k)


@pytest.mark.parametrize("make", [
    lambda: Repulsive({"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, RC, cutoff_width=DW),
    lambda: Buckingham(_BUCK, RC, cutoff_width=DW),
    lambda: BV(_SP, _PP, RC, form="power", cutoff_width=DW),
    lambda: BV(_SP, _PP, RC, form="exp",   cutoff_width=DW),
    lambda: BVV(_VSP, _PP, RC, form="power", cutoff_width=DW),
    lambda: BVV(_VSP, _PP, RC, form="exp",   cutoff_width=DW),
], ids=["rep", "buck", "bv-pow", "bv-exp", "bvv-pow", "bvv-exp"])
def test_fd_forces_with_switching(frame, make):
    pot = make()
    i, j, rv = pot._neighbors(frame.lattice, frame.positions, RC)
    r = np.linalg.norm(rv, axis=1)
    assert np.sum(r > RC - DW) > 100          # many pairs inside the switching shell
    _fd_check(pot, frame)


@pytest.mark.parametrize("make", [
    lambda: Repulsive({"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}, RC, cutoff_width=DW),
    lambda: Buckingham(_BUCK, RC, cutoff_width=DW),
], ids=["rep", "buck"])
def test_analytic_virial_at_original_cutoff(frame, make):
    """At r_c = 6.0 Å the hard-cutoff Buckingham virial disagreed with FD
    (pair 1.2e-5 Å from r_c); with switching the energy is C² and they agree."""
    pot = make()
    analytic = pot.stress(frame.lattice, frame.species, frame.positions)
    fd       = _fd_stress(pot, frame.lattice, frame.species, frame.positions)
    assert np.allclose(analytic, fd, atol=1e-7)
