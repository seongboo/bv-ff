"""
Uniform external electric field coupled to the rigid-ion charges.

E_field = −Σ_i q_i E·r_i = −E·d  (d = Σ q_i r_i, the cell dipole)

For a charge-neutral cell d is origin-independent, so the energy is well
defined up to the polarization-quantum branch (an ion crossing the periodic
boundary shifts d by q·L — the same caveat as any classical fixed-E method).
Forces are exactly f_i = q_i E (constant), which is what drives polarization
switching / hysteresis MD. Use through ``[extensions] efield = [Ex, Ey, Ez]``
(V/Å) in controls.toml — with charges in |e| and positions in Å the products
are directly eV.

This is an ANALYSIS term (switching, hysteresis, dielectric response), not a
fitting term: fitting against field-free DFT data with a field on would bias
every parameter.
"""
from __future__ import annotations

import numpy as np

from ..potentials import Potential


class EField(Potential):
    """Uniform field E (V/Å) acting on fixed point charges (|e|)."""

    def __init__(self, charges: dict[str, float], field):
        self.charges = charges
        self.field   = np.asarray(field, dtype=float).reshape(3)

    def _q(self, species) -> np.ndarray:
        return np.array([self.charges.get(s, 0.0) for s in species])

    def energy(self, lattice, species, positions) -> float:
        cart = self._frac_to_cart(np.asarray(lattice, dtype=float),
                                  np.asarray(positions, dtype=float))
        d = (self._q(species)[:, None] * cart).sum(axis=0)
        return -float(self.field @ d)

    def forces(self, lattice, species, positions) -> np.ndarray:
        return self._q(species)[:, None] * self.field[None, :]

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Under strain r → (I+ε)r the dipole transforms affinely, so
        ∂(−E·d)/∂ε_αβ = −E_α d_β; symmetrized as usual (only the symmetric
        part of ε is a strain)."""
        lattice = np.asarray(lattice, dtype=float)
        volume  = abs(np.linalg.det(lattice))
        cart = self._frac_to_cart(lattice, np.asarray(positions, dtype=float))
        d = (self._q(species)[:, None] * cart).sum(axis=0)
        sigma = -np.outer(self.field, d) / volume
        return 0.5 * (sigma + sigma.T)
