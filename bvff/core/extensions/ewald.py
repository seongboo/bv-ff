from __future__ import annotations

import numpy as np

from ..potentials import Potential, _LRUBytesCache, KE_COULOMB


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
# for the same rationale. Byte-bounded LRU for the same reason too: NPT
# MD produces a new lattice key every step.

KVEC_CACHE_MAX_BYTES = 64 * 1024 * 1024   # 64 MB

_KVEC_CACHE = _LRUBytesCache(KVEC_CACHE_MAX_BYTES)


def clear_kvec_cache() -> None:
    """Drop all cached reciprocal lattice vectors."""
    _KVEC_CACHE.clear()


def ewald_alpha(cutoff: float, accuracy: float = 1e-6) -> float:
    """Ewald splitting parameter alpha (1/Å) such that the real-space sum is
    converged at ``cutoff``: erfc(alpha·cutoff) ≈ accuracy ⇒
    alpha = sqrt(−ln accuracy)/cutoff. Depends only on the cutoff, not the cell."""
    return float(np.sqrt(-np.log(accuracy)) / float(cutoff))


def ewald_kmax(lattice: np.ndarray, alpha: float, accuracy: float = 1e-6) -> int:
    """Reciprocal index kmax such that the reciprocal sum is converged to
    ``accuracy`` for this cell: exp(−g²/4alpha²) ≈ accuracy ⇒
    g_cut = 2·alpha·sqrt(−ln accuracy), taking the largest per-direction index
    for an anisotropic cell. Clamped to [1, 12] so a pathological cell can't
    explode the (2kmax+1)³ k-grid — but a *binding* clamp means the reciprocal
    sum is NOT converged to ``accuracy``, so it warns instead of failing
    silently. A 2×2×2 supercell needs ~2× the kmax of the primitive cell (its
    reciprocal vectors are half as long), which is why kmax must be chosen per
    lattice for mixed-cell datasets."""
    lattice = np.asarray(lattice, dtype=float)
    g_cut   = 2.0 * alpha * np.sqrt(-np.log(accuracy))
    # |b_d| for each reciprocal lattice vector (rows of 2π·inv(L)ᵀ).
    recip   = 2.0 * np.pi * np.linalg.inv(lattice).T
    b_norms = np.linalg.norm(recip, axis=1)
    kmax    = int(np.ceil(g_cut / b_norms.min()))
    if kmax > 12:
        import warnings
        warnings.warn(
            f"Ewald kmax clamped to 12 (convergence to accuracy={accuracy:g} "
            f"needs kmax={kmax}): the reciprocal sum is under-converged for "
            f"this cell. Use a larger real-space cutoff (smaller alpha) or a "
            f"looser accuracy.",
            stacklevel=2,
        )
    return max(1, min(kmax, 12))


def ewald_parameters(
    lattice:  np.ndarray,
    cutoff:   float,
    accuracy: float = 1e-6,
) -> tuple[float, int]:
    """
    Choose the Ewald splitting parameter ``alpha`` (1/Å) and reciprocal index
    ``kmax`` from the cell, real-space cutoff, and a target accuracy — instead of
    hardcoding ``alpha=0.3, kmax=5`` regardless of cell size. See ``ewald_alpha``
    and ``ewald_kmax`` for the two criteria.

    Returns ``(alpha, kmax)``.
    """
    alpha = ewald_alpha(cutoff, accuracy)
    return alpha, ewald_kmax(lattice, alpha, accuracy)


class Ewald(Potential):
    """
    Ewald summation for long-range Coulomb interactions.

    E_total = E_real + E_recip + E_surface + E_self + E_background

    E_real       : short-range part in real space
    E_recip      : long-range part in reciprocal space
    E_surface    : dipole correction for polar systems (ferroelectrics)
    E_self       : self-interaction correction
    E_background : neutralizing-jellium correction, −KE·π·Q²/(2α²V). Zero for
                   a neutral cell; for a net charge Q it makes the total
                   alpha-independent (the standard uniform-background result)
                   instead of silently alpha-dependent.
    """

    # Conversion factor: 1 e^2/Angstrom = 14.3996 eV (shared with Coulomb).
    KE = KE_COULOMB

    def __init__(
        self,
        charges:  dict[str, float],  # {element: charge}
        alpha:    float,             # real/reciprocal space splitting parameter (1/Angstrom)
        kmax:     int | None,        # max k-vector index; None = per-lattice adaptive
        cutoff:   float,             # real-space cutoff (Angstrom)
        epsilon:  float = float("inf"),  # surface term dielectric: inf=tinfoil, 1.0=vacuum
        accuracy: float = 1e-6,      # target accuracy for adaptive kmax
    ):
        """
        Args:
            charges:  {element: charge} e.g. {"Pb": 1.38, "Ti": 0.99, "O": -0.79}
            alpha:    Ewald splitting parameter in 1/Angstrom
            kmax:     max reciprocal lattice vector index. ``None`` picks a
                      converged kmax per lattice via ``ewald_kmax`` — required
                      for datasets mixing cell sizes (a supercell needs a larger
                      kmax than the primitive cell for the same accuracy).
            cutoff:   real-space cutoff in Angstrom
            epsilon:  dielectric constant for the surface (dipole) term.
                      inf = tinfoil (term vanishes) — the standard choice for
                      bulk periodic crystals and the default (matching the
                      [extensions] ewald_epsilon controls default; an earlier
                      class default of 1.0 silently gave direct constructions
                      the vacuum boundary). 1.0 = vacuum, which penalizes any
                      uniformly polarized state and is discontinuous when an
                      atom wraps across the cell.
            accuracy: reciprocal-sum convergence target for adaptive kmax
        """
        self.charges  = charges
        self.alpha    = alpha
        self.kmax     = kmax
        self.cutoff   = cutoff
        self.epsilon  = epsilon
        self.accuracy = accuracy

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
        # ½ × full directed list (counts each interaction once; self-images,
        # which are part of the real-space Ewald sum, are handled correctly).
        r = np.linalg.norm(r_vecs, axis=1)
        e = np.sum(q[i_idx] * q[j_idx] * erfc(self.alpha * r) / r)
        return 0.5 * self.KE * float(e)

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
        r   = np.linalg.norm(r_vecs, axis=1)
        ar  = self.alpha * r
        fac = q[i_idx] * q[j_idx] * (
            erfc(ar) / r**3
            + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
        )
        # f_i = -dE/dr_i for r_vec = r_j - r_i (matches Coulomb sign convention).
        # Scatter to i over the full directed list; the (j,i) entry supplies j.
        df = -(self.KE * fac)[:, None] * r_vecs       # (M, 3)
        np.add.at(f, i_idx, df)
        return f

    # ──────────────────────────────────────────────
    # Reciprocal space term
    # ──────────────────────────────────────────────

    def _k_vectors(self, lattice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate reciprocal lattice vectors and their squared magnitudes.

        With ``kmax=None`` the index is chosen per lattice (``ewald_kmax``), so
        mixed-cell datasets get a converged reciprocal sum for every frame.
        Result depends only on (lattice, kmax) and is cached content-keyed.
        Both arrays are read-only outputs and shared across callers.
        """
        lattice = np.asarray(lattice, dtype=float)
        kmax = self.kmax if self.kmax is not None else ewald_kmax(
            lattice, self.alpha, self.accuracy
        )
        key = (lattice.tobytes(), kmax)
        cached = _KVEC_CACHE.get(key)
        if cached is not None:
            return cached

        recip = 2 * np.pi * np.linalg.inv(lattice).T          # (3, 3)
        rng   = np.arange(-kmax, kmax + 1)
        hh, kk, ll = np.meshgrid(rng, rng, rng, indexing="ij")
        hkl    = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1)  # ((2k+1)^3, 3)
        # Drop the (0,0,0) row
        nonzero = np.any(hkl != 0, axis=1)
        hkl    = hkl[nonzero]
        k_vecs = hkl @ recip                                  # (K, 3)
        k2     = np.einsum("ij,ij->i", k_vecs, k_vecs)        # (K,)
        _KVEC_CACHE.put(key, (k_vecs, k2), k_vecs.nbytes + k2.nbytes + len(key[0]))
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
    # Neutralizing background term
    # ──────────────────────────────────────────────

    def _energy_background(self, q: np.ndarray, volume: float) -> float:
        """Uniform neutralizing-background (jellium) correction for a cell
        with net charge Q: E = −KE·π·Q²/(2α²V). Exactly zero for a neutral
        cell; without it a non-neutral Ewald energy depends on the arbitrary
        splitting parameter alpha. Position-independent → zero forces."""
        Q = float(np.sum(q))
        return -self.KE * np.pi * Q * Q / (2.0 * self.alpha**2 * volume)

    # ──────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────

    def energy(self, lattice, species, positions) -> float:
        cart   = self._frac_to_cart(lattice, positions)
        q      = self._get_charges(species)
        volume = float(np.abs(np.linalg.det(np.asarray(lattice, dtype=float))))
        return (
            self._energy_real(lattice, positions, q)
            + self._energy_recip(cart, lattice, q)
            + self._energy_surface(cart, lattice, q)
            + self._energy_self(q)
            + self._energy_background(q, volume)
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
            r    = np.linalg.norm(r_vecs, axis=1)
            ar   = self.alpha * r
            qiqj = q[i_idx] * q[j_idx]
            erfc_ar = erfc(ar)
            # ½ × full directed list for energy; scatter-to-i for forces.
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

        # Surface, self, and background terms (cheap, just delegate).
        e_surf = self._energy_surface(cart, lattice, q)
        f_surf = self._forces_surface(cart, lattice, q, n)
        e_self = self._energy_self(q)
        e_bg   = self._energy_background(q, float(volume))

        return e_real + e_recip + e_surf + e_self + e_bg, f_real + f_recip + f_surf

    # ──────────────────────────────────────────────
    # Analytic virial stress
    # ──────────────────────────────────────────────

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic Ewald virial σ_αβ = (1/V) ∂E/∂ε_αβ (replaces the 18-energy-
        evaluation FD fallback — the dominant cost of stress-fitting).

        Per component (validated against the FD path in the tests):
          real:    pairwise, σ = (1/2V) Σ_directed df ⊗ r (erfc kernel).
          recip:   with E_k = KE·(2π/V)·A(k)|S(k)|², A = e^(−k²/4α²)/k²,
                   strain moves k → (I−εᵀ)k and V → V(1+trε) while k·r (and
                   hence S(k)) is invariant, giving
                   σ = (1/V) Σ_k E_k [2(1/(4α²) + 1/k²) k_αk_β − δ_αβ].
          surface: d = Σqr transforms affinely,
                   σ = (2·KE·c/V²) d⊗d − (E_surf/V)·I, c = 2π/(2ε+1)
                   (zero under the tinfoil default).
          self:    strain-independent → 0.
          background: E ∝ 1/V → σ = −(E_bg/V)·I.
        """
        from scipy.special import erfc
        lattice = np.asarray(lattice, dtype=float)
        volume  = float(np.abs(np.linalg.det(lattice)))
        cart    = self._frac_to_cart(lattice, positions)
        q       = self._get_charges(species)
        sigma   = np.zeros((3, 3))

        # Real-space part (pairwise virial, same df as _forces_real).
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) > 0:
            r   = np.linalg.norm(r_vecs, axis=1)
            ar  = self.alpha * r
            fac = q[i_idx] * q[j_idx] * (
                erfc(ar) / r**3
                + 2 * self.alpha / np.sqrt(np.pi) * np.exp(-ar * ar) / r**2
            )
            df = -(self.KE * fac)[:, None] * r_vecs
            sigma += 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume

        # Reciprocal part.
        k_vecs, k2  = self._k_vectors(lattice)
        phase       = cart @ k_vecs.T
        S_real      = q @ np.cos(phase)
        S_imag      = q @ np.sin(phase)
        exp_fac     = np.exp(-k2 / (4 * self.alpha**2)) / k2
        E_k         = self.KE * (2 * np.pi / volume) * exp_fac * (S_real**2 + S_imag**2)
        coeff       = 2.0 * (1.0 / (4.0 * self.alpha**2) + 1.0 / k2)
        sigma += (
            np.einsum("m,ma,mb->ab", E_k * coeff, k_vecs, k_vecs)
            - np.eye(3) * float(np.sum(E_k))
        ) / volume

        # Surface (dipole) part — vanishes under the tinfoil boundary.
        if not np.isinf(self.epsilon):
            dipole = np.sum(q[:, np.newaxis] * cart, axis=0)
            c      = 2 * np.pi / (2 * self.epsilon + 1)
            e_surf = self.KE * c * float(np.dot(dipole, dipole)) / volume
            sigma += (2.0 * self.KE * c / volume**2) * np.outer(dipole, dipole)
            sigma -= np.eye(3) * (e_surf / volume)

        # Self term: no strain dependence. Background: E ∝ 1/V.
        e_bg = self._energy_background(q, volume)
        sigma -= np.eye(3) * (e_bg / volume)

        return 0.5 * (sigma + sigma.T)
