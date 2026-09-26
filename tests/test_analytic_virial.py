"""
Analytic virials of BV, BVV and Ewald vs the finite-difference strain
derivative (Potential.stress in the base class), on a triclinically strained
frame so that all nine components are exercised.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parsers.dataset import load_dataset, DatasetEntry
from src.potentials  import Potential, BV, BVV, clear_neighbor_cache
from src.extensions.ewald import Ewald, clear_kvec_cache


_ROOT = Path(__file__).resolve().parent.parent
_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.3}, "O": {"V0": 2.0, "S": 0.7}}
_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0, "b": 0.37}, "O-Ti": {"r0": 1.81, "C": 5.2, "b": 0.40}}
_VSP  = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.2}, "O": {"W0": 0.1, "D": 0.05}}
_Q    = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}


@pytest.fixture(scope="module")
def tri_frame():
    clear_neighbor_cache(); clear_kvec_cache()
    fr = load_dataset(entries=[DatasetEntry(
        path=str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start=100, frame_end=101, stride=1,
    )]).frames[0]
    shear = np.array([[1.00, 0.04, -0.03],
                      [0.02, 0.99,  0.05],
                      [-0.04, 0.01, 1.02]])
    return fr.lattice @ shear.T, fr.species, fr.positions


TERMS = {
    "bv-pow":       lambda: BV(_SP, _PP, 6.0, form="power"),
    "bv-exp-sw":    lambda: BV(_SP, _PP, 6.0, form="exp", cutoff_width=1.0),
    "bv-pow-sw":    lambda: BV(_SP, _PP, 6.0, form="power", cutoff_width=1.0),
    "bvv-pow":      lambda: BVV(_VSP, _PP, 6.0, form="power"),
    "bvv-pow-sw":   lambda: BVV(_VSP, _PP, 6.0, form="power", cutoff_width=1.0),
    "bvv-exp-sw":   lambda: BVV(_VSP, _PP, 6.0, form="exp", cutoff_width=1.0),
    "ewald":        lambda: Ewald(_Q, cutoff=6.0, accuracy=1e-10),
    "ewald-vacuum": lambda: Ewald(_Q, cutoff=6.0, accuracy=1e-10, epsilon=1.0),
}


@pytest.mark.filterwarnings("ignore:Ewald with finite epsilon")
@pytest.mark.parametrize("name", list(TERMS))
def test_analytic_virial_matches_fd(tri_frame, name):
    pot = TERMS[name]()
    L, sp, x = tri_frame
    an = pot.stress(L, sp, x)
    fd = Potential.stress(pot, L, sp, x, eps=1e-5)          # base-class FD path
    scale = max(np.abs(fd).max(), 1e-6)
    np.testing.assert_allclose(an, fd, atol=2e-6 * scale + 1e-10, err_msg=name)
    np.testing.assert_allclose(an, an.T, atol=1e-12 * scale + 1e-14)


@pytest.mark.parametrize("name", ["bv-pow-sw", "bvv-pow-sw", "ewald"])
def test_virial_translation_invariant(tri_frame, name):
    pot = TERMS[name]()
    L, sp, x = tri_frame
    s0 = pot.stress(L, sp, x)
    s1 = pot.stress(L, sp, x + np.array([0.137, -0.29, 0.61]))
    np.testing.assert_allclose(s1, s0, atol=1e-10)
