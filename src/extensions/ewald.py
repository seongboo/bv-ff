from __future__ import annotations

import numpy as np

from ..potentials import Potential


# ──────────────────────────────────────────────
# k-vector cache
# ──────────────────────────────────────────────
#
# k-vectors and their squared magnitudes depend only on the lattice and
# kmax, neither of which changes during fitting. Memoize so each frame
# computes the (2*kmax+1)^3 - 1 vectors exactly once across all SA
# iterations and across energy/forces calls.

# Key is (lattice.tobytes(), kmax) — content-based, so identical lattices
# share a cache entry and freshly-allocated arrays (with reused memory
# addresses after GC) don't cause stale hits. See potentials._NEIGHBOR_CACHE
# for the same rationale.
_KVEC_CACHE: dict[
    tuple[bytes, int],
    tuple[np.ndarray, np.ndarray],
] = {}


def clear_kvec_cache() -> None:
    """Drop all cached reciprocal lattice vectors."""
    _KVEC_CACHE.clear()


class Ewald(Potential):
    """
    Ewald summation for long-range Coulomb interactions.

    E_total = E_real + E_recip + E_surface + E_self

    E_real    : short-range part in real space
    E_recip   : long-range part in reciprocal space
    E_surface : dipole correction for polar systems (ferroelectrics)
    E_self    : self-interaction correction
    """

    # Conversion factor: 1 e^2/Angstrom = 14.3996 eV
    KE = 14.3996

    def __init__(
        self,
        charges:  dict[str, float],  # {element: charge}
        alpha:    float,             # real/reciprocal space splitting parameter (1/Angstrom)
        kmax:     int,               # max k-vector index in each direction
        cutoff:   float,             # real-space cutoff (Angstrom)
        epsilon:  float = 1.0,       # surface term dielectric: 1.0=vacuum, inf=tinfoil
    ):
        """
        Args:
            charges:  {element: charge} e.g. {"Pb": 1.38, "Ti": 0.99, "O": -0.79}
            alpha:    Ewald splitting parameter in 1/Angstrom
            kmax:     max reciprocal lattice vector index
            cutoff:   real-space cutoff in Angstrom
            epsilon:  dielectric constant for surface term (1.0 = vacuum boundary)
        """
        self.charges = charges
        self.alpha   = alpha
        self.kmax    = kmax
        self.cutoff  = cutoff
        self.epsilon = epsilon

    def _get_charges(self, species: list[str]) -> np.ndarray:
        return np.array([self.charges.get(s, 0.0) for s in species])

    # ──────────────────────────────────────────────
    # Real space term
    # ──────────────────────────────────────────────

    def _energy_real(
        self,
        lattice:   np.ndarray,  # (3, 3)
        positions: np.ndarray,  # (N, 3) fractional
        q:         np.ndarray,  # (N,)
    ) -> float:
        from scipy.special import erfc
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        mask  = i_idx < j_idx
        i, j  = i_idx[mask], j_idx[mask]
        r     = np.linalg.norm(r_vecs[mask], axis=1)
        e     = np.sum(q[i] * q[j] * erfc(self.alpha * r) / r)
        return self.KE * float(e)

    def _forces_real(
        self,
        lattice:   np.ndarray,
        positions: np.ndarray,
        q:         np.ndarray,
        n:         int,
    ) -> np.ndarray:
        from scipy.special import erfc
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        # Iterate each pair once (i<j) so f[i]+=df / f[j]-=df is not doubled
        # by the symmetric (j,i) entry the neighbor list also emits.
        mask = i_idx < j_idx
        if not mask.any():
            return f
        i, j = i_idx[mask], j_idx[mask]
        rv   = r_vecs[mask]
        r    = np.linalg.norm(rv, axis=1)
        ar   = self.alpha * r
        fac  = q[i] * q[j] * (
            erfc(ar) / r**3
            + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
        )
        # f_i = -dE/dr_i for r_vec = r_j - r_i (matches Coulomb sign convention).
        df   = -(self.KE * fac)[:, None] * rv         # (M, 3)
        np.add.at(f, i,  df)
        np.add.at(f, j, -df)
        return f

    # ──────────────────────────────────────────────
    # Reciprocal space term
    # ──────────────────────────────────────────────

    def _k_vectors(self, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate reciprocal lattice vectors and their squared magnitudes.

        Result depends only on (lattice, kmax) so we cache by (id(lattice), kmax).
        Both arrays are read-only outputs and shared across callers.
        """
        key = (lattice.tobytes(), self.kmax)
        cached = _KVEC_CACHE.get(key)
        if cached is not None:
            return cached

        recip = 2 * np.pi * np.linalg.inv(lattice).T          # (3, 3)
        rng   = np.arange(-self.kmax, self.kmax + 1)
        hh, kk, ll = np.meshgrid(rng, rng, rng, indexing="ij")
        hkl    = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1)  # ((2k+1)^3, 3)
        # Drop the (0,0,0) row
        nonzero = np.any(hkl != 0, axis=1)
        hkl    = hkl[nonzero]
        k_vecs = hkl @ recip                                  # (K, 3)
        k2     = np.einsum("ij,ij->i", k_vecs, k_vecs)        # (K,)
        _KVEC_CACHE[key] = (k_vecs, k2)
        return k_vecs, k2

    def _energy_recip(
        self,
        cart:    np.ndarray,
        lattice: np.ndarray,
        q:       np.ndarray,
    ) -> float:
        volume      = np.abs(np.linalg.det(lattice))
        k_vecs, k2  = self._k_vectors(lattice)             # (K, 3), (K,)
        phase       = cart @ k_vecs.T                       # (N, K)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        S_real      = q @ cos_p                             # (K,)
        S_imag      = q @ sin_p                             # (K,)
        exp_fac     = np.exp(-k2 / (4 * self.alpha**2)) / k2
        e           = np.sum(exp_fac * (S_real**2 + S_imag**2))
        return self.KE * (4 * np.pi / (2 * volume)) * float(e)

    def _forces_recip(
        self,
        cart:    np.ndarray,
        lattice: np.ndarray,
        q:       np.ndarray,
        n:       int,
    ) -> np.ndarray:
        volume      = np.abs(np.linalg.det(lattice))
        k_vecs, k2  = self._k_vectors(lattice)              # (K, 3), (K,)
        phase       = cart @ k_vecs.T                        # (N, K)
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        S_real      = q @ cos_p                              # (K,)
        S_imag      = q @ sin_p                              # (K,)
        exp_fac     = np.exp(-k2 / (4 * self.alpha**2)) / k2 # (K,)
        # per (i, k):  exp_fac[k] * q[i] * (-S_real[k]*sin[i,k] + S_imag[k]*cos[i,k])
        weight      = exp_fac * (-S_real * sin_p + S_imag * cos_p)  # (N, K)
        weight     *= q[:, None]
        f           = weight @ k_vecs                        # (N, 3)
        # f = -dE_recip/dr. The bracket above is +dE/dr (per the 2*S*dS/dr
        # chain rule); we negate by absorbing the minus into the prefactor.
        f          *= -self.KE * (8 * np.pi / (2 * volume))
        return f

    # ──────────────────────────────────────────────
    # Surface term (dipole correction)
    # ──────────────────────────────────────────────

    def _energy_surface(
        self,
        cart:    np.ndarray,
        lattice: np.ndarray,
        q:       np.ndarray,
    ) -> float:
        if np.isinf(self.epsilon):
            return 0.0
        volume  = np.abs(np.linalg.det(lattice))
        dipole  = np.sum(q[:, np.newaxis] * cart, axis=0)  # (3,)
        fac     = 2 * np.pi / ((2 * self.epsilon + 1) * volume)
        return self.KE * fac * np.dot(dipole, dipole)

    def _forces_surface(
        self,
        cart:    np.ndarray,
        lattice: np.ndarray,
        q:       np.ndarray,
        n:       int,
    ) -> np.ndarray:
        if np.isinf(self.epsilon):
            return np.zeros((n, 3))
        volume  = np.abs(np.linalg.det(lattice))
        dipole  = np.sum(q[:, np.newaxis] * cart, axis=0)   # (3,)
        fac     = 4 * np.pi / ((2 * self.epsilon + 1) * volume)
        return -self.KE * fac * q[:, None] * dipole[None, :]

    # ──────────────────────────────────────────────
    # Self energy term
    # ──────────────────────────────────────────────

    def _energy_self(self, q: np.ndarray) -> float:
        return -self.KE * self.alpha / np.sqrt(np.pi) * np.sum(q**2)

    # ──────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────

    def energy(self, lattice, species, positions) -> float:
        cart = self._frac_to_cart(lattice, positions)
        q    = self._get_charges(species)
        return (
            self._energy_real(lattice, positions, q)
            + self._energy_recip(cart, lattice, q)
            + self._energy_surface(cart, lattice, q)
            + self._energy_self(q)
        )

    def forces(self, lattice, species, positions) -> np.ndarray:
        n    = len(species)
        cart = self._frac_to_cart(lattice, positions)
        q    = self._get_charges(species)
        return (
            self._forces_real(lattice, positions, q, n)
            + self._forces_recip(cart, lattice, q, n)
            + self._forces_surface(cart, lattice, q, n)
        )

    def energy_and_forces(self, lattice, species, positions):
        """Single-pass Ewald — the reciprocal term builds (phase, cos, sin,
        S_real, S_imag, exp_fac) once and reuses them for both the energy
        sum and the per-atom force gradient. (Real-space and surface terms
        are cheap; default fallback would also work for them.)
        """
        n     = len(species)
        cart  = self._frac_to_cart(lattice, positions)
        q     = self._get_charges(species)

        # Real-space: same neighbour list, same fac/df scaffolding.
        from scipy.special import erfc
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        e_real = 0.0
        f_real = np.zeros((n, 3))
        if len(i_idx) > 0:
            mask = i_idx < j_idx
            if mask.any():
                i, j = i_idx[mask], j_idx[mask]
                rv   = r_vecs[mask]
                r    = np.linalg.norm(rv, axis=1)
                ar   = self.alpha * r
                qiqj = q[i] * q[j]
                erfc_ar = erfc(ar)
                e_real  = float(self.KE * np.sum(qiqj * erfc_ar / r))
                fac     = qiqj * (
                    erfc_ar / r**3
                    + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
                )
                df      = -(self.KE * fac)[:, None] * rv
                np.add.at(f_real, i,  df)
                np.add.at(f_real, j, -df)

        # Reciprocal-space: share phase / cos / sin / S between energy & forces.
        volume      = np.abs(np.linalg.det(lattice))
        k_vecs, k2  = self._k_vectors(lattice)
        phase       = cart @ k_vecs.T
        cos_p, sin_p = np.cos(phase), np.sin(phase)
        S_real      = q @ cos_p
        S_imag      = q @ sin_p
        exp_fac     = np.exp(-k2 / (4 * self.alpha**2)) / k2

        e_recip = self.KE * (4 * np.pi / (2 * volume)) * float(
            np.sum(exp_fac * (S_real**2 + S_imag**2))
        )
        weight  = exp_fac * (-S_real * sin_p + S_imag * cos_p)
        weight *= q[:, None]
        f_recip = weight @ k_vecs
        f_recip *= -self.KE * (8 * np.pi / (2 * volume))

        # Surface and self terms (cheap, just delegate).
        e_surf = self._energy_surface(cart, lattice, q)
        f_surf = self._forces_surface(cart, lattice, q, n)
        e_self = self._energy_self(q)

        return e_real + e_recip + e_surf + e_self, f_real + f_recip + f_surf
