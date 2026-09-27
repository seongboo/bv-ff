"""
ASE Calculator wrapper around a fitted BVFF.

Exposing the bond-valence force field as an ``ase.calculators.calculator.Calculator``
unlocks the whole ASE ecosystem — geometry relaxation, molecular dynamics,
phonons, equation-of-state — which is what turns a fitted potential into a
*validated ferroelectric* model (double-well, spontaneous polarization, Tc).

Conventions (verified in tests/test_calculator.py):
  - ``BVFF.forces`` already returns the physical force f = -∂E/∂r (Cartesian,
    eV/Å), so it maps to ASE 'forces' with no sign flip.
  - ``BVFF.stress`` returns σ = (1/V) ∂E/∂ε (eV/Å³, 3×3), the same definition
    ASE uses; it is converted to the Voigt 6-vector ASE expects.
  - ``BVFF`` works in fractional coordinates with lattice rows = cell vectors,
    matching ``atoms.cell`` / ``atoms.get_scaled_positions()``.
"""
from __future__ import annotations

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .potentials import BVFF


class BVFFCalculator(Calculator):
    """ASE calculator backed by a fitted :class:`~src.potentials.BVFF`."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, bvff: BVFF, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.bvff = bvff

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        at      = self.atoms
        lattice = np.asarray(at.cell.array, dtype=float)
        species = at.get_chemical_symbols()
        frac    = at.get_scaled_positions(wrap=False)

        energy, forces = self.bvff.energy_and_forces(lattice, species, frac)
        self.results["energy"]      = float(energy)
        self.results["free_energy"] = float(energy)
        self.results["forces"]      = np.asarray(forces, dtype=float)

        if "stress" in properties:
            from ase.stress import full_3x3_to_voigt_6_stress
            sigma = self.bvff.stress(lattice, species, frac)   # (3,3) eV/Å³, (1/V)∂E/∂ε
            self.results["stress"] = full_3x3_to_voigt_6_stress(np.asarray(sigma, dtype=float))


# ──────────────────────────────────────────────
# Convenience builders
# ──────────────────────────────────────────────

def calculator_from_params(controls, params) -> BVFFCalculator:
    """Build a :class:`BVFFCalculator` from parsed controls + parameters.
    Ewald parameters adapt to each cell automatically (see ``build_bvff``)."""
    from .main import build_bvff
    return BVFFCalculator(build_bvff(controls, params))


def atoms_from_frame(frame):
    """Build an ``ase.Atoms`` (with periodic cell) from a dataset ``Frame``."""
    from ase import Atoms
    return Atoms(
        symbols         = frame.species,
        scaled_positions= np.asarray(frame.positions, dtype=float),
        cell            = np.asarray(frame.lattice, dtype=float),
        pbc             = True,
    )
