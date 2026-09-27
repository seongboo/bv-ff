"""
BVFFCalculator (ASE interface) correctness.

Validates that the ASE wrapper reports the same energy as the underlying BVFF,
and — crucially — that the force and stress *sign conventions* match ASE's own
numerical references (``calculate_numerical_forces`` / ``calculate_numerical_stress``),
which use the calculator's energy and therefore define the sign ASE expects.
"""
from __future__ import annotations

import numpy as np
import pytest

from parsers.dataset   import load_dataset, DatasetEntry
from src.potentials    import Coulomb, Repulsive, BV, BVV, BVFF, clear_neighbor_cache
from src.calculator    import BVFFCalculator, atoms_from_frame


CUTOFF = 6.0

_CHARGES = {"Pb": 1.4, "Ti": 1.0, "O": -0.8}
_B       = {"O-O": 1.83, "O-Pb": 2.17, "O-Ti": 1.28}
_BV_SP   = {"Pb": {"V0": 2.0, "S": 0.5}, "Ti": {"V0": 4.0, "S": 0.5}, "O": {"V0": 2.0, "S": 0.5}}
_BV_PP   = {"O-Pb": {"r0": 2.06, "C": 6.0}, "O-Ti": {"r0": 1.81, "C": 5.2}}
_BVV_SP  = {"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}, "O": {"W0": 0.0, "D": 0.0}}


def _bvff():
    return BVFF([
        Coulomb(_CHARGES, CUTOFF),
        Repulsive(_B, CUTOFF),
        BV(_BV_SP, _BV_PP, CUTOFF),
        BVV(_BVV_SP, _BV_PP, CUTOFF),
    ])


@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    data = load_dataset(entries=[DatasetEntry(
        path="examples/PbTiO3/pbtio3_222_300K.parquet",
        frame_start=100, frame_end=101, stride=1,
    )])
    return data.frames[0]


def test_energy_matches_bvff(frame):
    bvff  = _bvff()
    atoms = atoms_from_frame(frame)
    atoms.calc = BVFFCalculator(bvff)
    e_ase  = atoms.get_potential_energy()
    e_bvff = bvff.energy(frame.lattice, frame.species, frame.positions)
    assert e_ase == pytest.approx(e_bvff, abs=1e-10)


def test_forces_match_bvff(frame):
    bvff  = _bvff()
    atoms = atoms_from_frame(frame)
    atoms.calc = BVFFCalculator(bvff)
    f_ase  = atoms.get_forces()
    f_bvff = bvff.forces(frame.lattice, frame.species, frame.positions)
    np.testing.assert_allclose(f_ase, f_bvff, atol=1e-10)


def test_forces_match_ase_numerical(frame):
    """ASE numerical forces use the calculator's energy → validates the sign."""
    atoms = atoms_from_frame(frame)
    atoms.calc = BVFFCalculator(_bvff())
    f_analytic  = atoms.get_forces()
    f_numerical = atoms.calc.calculate_numerical_forces(atoms, d=1e-4)
    np.testing.assert_allclose(f_analytic, f_numerical, atol=1e-4)


def test_stress_matches_ase_numerical(frame):
    """Stress Voigt order + sign must match ASE's numerical strain derivative."""
    atoms = atoms_from_frame(frame)
    atoms.calc = BVFFCalculator(_bvff())
    s_analytic  = atoms.get_stress()
    s_numerical = atoms.calc.calculate_numerical_stress(atoms, d=1e-5)
    np.testing.assert_allclose(s_analytic, s_numerical, atol=1e-5)
