from __future__ import annotations

import warnings

import numpy as np

from ..potentials import Potential, COULOMB_CONSTANT


# ──────────────────────────────────────────────
# k-vector cache
# ──────────────────────────────────────────────
#
# k-vectors and their squared magnitudes depend only on the lattice and the
# reciprocal cutoff k_c, neither of which changes during fitting. Memoize so
# each frame builds its k-set exactly once across all SA iterations and
# across energy/forces calls.

# Key is (lattice.tobytes(), k_c) — content-based, so identical lattices
# share a cache entry and freshly-allocated arrays (with reused memory
# addresses after GC) don't cause stale hits. See potentials._NEIGHBOR_CACHE
# for the same rationale.
_KVEC_CACHE: dict[
    tuple[bytes, float],
    tuple[np.ndarray, np.ndarray],
] = {}


def clear_kvec_cache() -> None:
    """Drop all cached reciprocal lattice vectors."""
    _KVEC_CACHE.clear()


class Ewald(Potential):
    """
    Ewald summation for long-range Coulomb interactions (k_e = COULOMB_CONSTANT).

        E = E_real + E_recip + E_self (+ E_surface for finite epsilon)

        E_real  = (k_e/2) Σ'_(i,j,n) q_i q_j erfc(α r)/r
        E_recip = (2π k_e / V) Σ_{0<|k|<=k_c} exp(-k²/4α²)/k² |S(k)|²,  S(k) = Σ_j q_j e^{ik·r_j}
        E_self  = -k_e (α/√π) Σ_i q_i²

    Parameters from a target accuracy δ. Truncation errors of both sums are
    balanced by requiring exp(-α² r_c²) = exp(-k_c²/4α²) = δ, i.e.

        p = √(-ln δ),   α = p / r_c,   k_c = 2 α p.

    The k-sum is cut on |k| <= k_c (a sphere), not on integer indices, so a
    given (r_c, δ) gives the same accuracy for any cell size or shape
    (supercells, NPT, mixed datasets). ``alpha`` / ``kcut`` may be overridden
    explicitly (e.g. to test α-independence).

    Boundary condition. The default epsilon = inf is the tin-foil (conducting)
    boundary: no surface term and no depolarising field — the appropriate
    choice for bulk (short-circuited) ferroelectrics and the LAMMPS
    ``kspace_style ewald`` default. A finite epsilon adds
    2π k_e |M|² / ((2ε+1) V) with M = Σ q_i r_i; M is not well defined under
    PBC (it jumps by q·L when an atom is wrapped), so this is only meaningful
    for unwrapped trajectories and triggers a warning.

    The cell must be charge-neutral: otherwise the k = 0 term diverges and
    the energy depends on α. A non-neutral charge set raises ValueError.
    """

    KE = COULOMB_CONSTANT

    def __init__(
        self,
        charges:  dict[str, float],  # {element: charge}
        cutoff:   float,             # real-space cutoff r_c (Angstrom)
        accuracy: float = 1e-6,      # target truncation accuracy δ (0 < δ < 1)
        epsilon:  float = np.inf,    # surface dielectric: inf = tin-foil
        alpha:    float | None = None,   # override α (1/Angstrom)
        kcut:     float | None = None,   # override k_c (1/Angstrom)
    ):
        if cutoff <= 0:
            raise ValueError(f"Ewald cutoff must be > 0, got {cutoff}.")
        if not (0.0 < accuracy < 1.0):
            raise ValueError(f"Ewald accuracy must be in (0, 1), got {accuracy}.")
        p = np.sqrt(-np.log(accuracy))
        self.charges  = charges
        self.cutoff   = float(cutoff)
        self.accuracy = float(accuracy)
        self.alpha    = float(alpha) if alpha is not None else p / self.cutoff
        self.kcut     = float(kcut)  if kcut  is not None else 2.0 * self.alpha * p
        self.epsilon  = epsilon
        if not np.isinf(epsilon):
            warnings.warn(
                "Ewald with finite epsilon adds a dipole surface term that is not "
                "invariant under wrapping atoms into the cell; use epsilon=inf "
                "(tin-foil) for periodic bulk systems.",
                stacklevel=2,
            )

    def _get_charges(self, species: list[str]) -> np.ndarray:
        q = np.array([self.charges.get(s, 0.0) for s in species], dtype=float)
        total = float(q.sum())
        if abs(total) > 1e-8 * max(1.0, float(np.abs(q).sum())):
            raise ValueError(
                f"Ewald requires a charge-neutral cell; net charge = {total:+.6e} e "
                f"(charges {self.charges}). Adjust charges so Σ n_s q_s = 0."
            )
        return q

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
        # Full ordered list (all images) → factor ½.
        r     = np.linalg.norm(r_vecs, axis=1)
        e     = 0.5 * np.sum(q[i_idx] * q[j_idx] * erfc(self.alpha * r) / r)
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
        # Full ordered list: force on i = sum over entries (i, j, n).
        r    = np.linalg.norm(r_vecs, axis=1)
        ar   = self.alpha * r
        fac  = q[i_idx] * q[j_idx] * (
            erfc(ar) / r**3
            + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
        )
        # f_i = -dE/dr_i for r_vec = r_j + n·L - r_i (matches Coulomb sign convention).
        df   = -(self.KE * fac)[:, None] * r_vecs     # (M, 3)
        np.add.at(f, i_idx, df)
        return f

    # ──────────────────────────────────────────────
    # Reciprocal space term
    # ──────────────────────────────────────────────

    def _k_vectors(self, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        All reciprocal vectors k = 2π (h, k, l)·B with 0 < |k| <= k_c, where
        the rows of B = (L⁻¹)ᵀ are the reciprocal vectors b_i (a_i·b_j = δ_ij).

        Index range. Since k·a_i = 2π h_i, the projection onto â_i gives
        |k| >= 2π |h_i| / |a_i|, so |k| <= k_c requires
            |h_i| <= floor(k_c |a_i| / 2π),
        the reciprocal-space analogue of the real-space image bound.

        Cached on (lattice bytes, k_c). Returned arrays are shared read-only.
        """
        key = (lattice.tobytes(), self.kcut)
        cached = _KVEC_CACHE.get(key)
        if cached is not None:
            return cached

        lattice = np.asarray(lattice, dtype=float)
        recip   = 2 * np.pi * np.linalg.inv(lattice).T          # (3, 3) rows = 2π b_i
        h_max   = np.floor(self.kcut * np.linalg.norm(lattice, axis=1) / (2 * np.pi)).astype(int)
        rng     = [np.arange(-m, m + 1) for m in h_max]
        hkl     = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)
        k_vecs  = hkl @ recip                                   # (K, 3)
        k2      = np.einsum("ij,ij->i", k_vecs, k_vecs)         # (K,)
        keep    = (k2 > 0.0) & (k2 <= self.kcut**2)
        k_vecs, k2 = k_vecs[keep], k2[keep]
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
            # Full ordered list (all images): ½ on the energy, force on i
            # accumulated from entries (i, j, n) only.
            r       = np.linalg.norm(r_vecs, axis=1)
            ar      = self.alpha * r
            qiqj    = q[i_idx] * q[j_idx]
            erfc_ar = erfc(ar)
            e_real  = float(0.5 * self.KE * np.sum(qiqj * erfc_ar / r))
            fac     = qiqj * (
                erfc_ar / r**3
                + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
            )
            df      = -(self.KE * fac)[:, None] * r_vecs
            np.add.at(f_real, i_idx, df)

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
