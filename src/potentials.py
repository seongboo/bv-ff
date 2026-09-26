from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# Unit conversion for the virial stress. Internally stress is computed in
# eV/Å³ (σ = (1/V) ∂E/∂ε). VASP / vasprun.xml report stress in kBar, so a
# fit against DFT stress needs this factor. 1 eV/Å³ = 160.21766208 GPa
# = 1602.1766208 kBar.
EV_PER_ANG3_TO_KBAR = 1602.1766208

# Coulomb constant e²/(4πε₀) in eV·Å, so that k_e q_i q_j / r is in eV for
# charges in units of e and r in Å. Shared by Coulomb and Ewald.
COULOMB_CONSTANT = 14.3996454784


# ──────────────────────────────────────────────
# Neighbor list cache
# ──────────────────────────────────────────────
#
# During fitting, each Frame's lattice/positions are reused across many
# loss evaluations and across every Potential term (Coulomb, Repulsive,
# BV, BVV, Angle, Ewald-real). The neighbor list (i_idx, j_idx, r_vecs)
# depends only on (positions, lattice, cutoff) — not on parameters — so
# we compute it once per (frame, cutoff) and reuse it everywhere.
#
# Cache key uses the raw byte representation of (positions, lattice) so that
# two arrays with identical contents collide regardless of Python object
# identity. An earlier id()-keyed version produced subtle stale-hit bugs
# when CPython reused the memory of a GC'd array (e.g. during a finite-
# difference test that allocates many short-lived perturbations). tobytes()
# costs ~1µs for a 40-atom system — negligible vs the saved N×N PBC loop.
# Cached tuples must be treated as read-only (no caller mutates them).

_NEIGHBOR_CACHE: dict[
    tuple[bytes, bytes, float],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


def clear_neighbor_cache() -> None:
    """Drop all cached neighbor lists. Call if frame arrays are mutated in place."""
    _NEIGHBOR_CACHE.clear()


# ──────────────────────────────────────────────
# Abstract base class
# ──────────────────────────────────────────────

class Potential(ABC):
    """Abstract base class for all BVFF potential terms."""

    @abstractmethod
    def energy(
        self,
        lattice:   np.ndarray,  # (3, 3) lattice vectors in Angstrom
        species:   list[str],   # element symbols, length = N
        positions: np.ndarray,  # (N, 3) fractional coordinates
    ) -> float:
        """Compute potential energy in eV."""
        ...

    @abstractmethod
    def forces(
        self,
        lattice:   np.ndarray,  # (3, 3)
        species:   list[str],
        positions: np.ndarray,  # (N, 3)
    ) -> np.ndarray:
        """Compute forces in eV/Angstrom. Returns (N, 3) array."""
        ...

    def energy_and_forces(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute energy and forces in a single call. Default implementation
        defers to the separate ``energy`` and ``forces`` methods; subclasses
        may override to share intermediate arrays (r, r0, C, Vij, …) between
        the two computations.
        """
        return (
            self.energy(lattice, species, positions),
            self.forces(lattice, species, positions),
        )

    def stress(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
        eps:       float = 1e-4,
    ) -> np.ndarray:
        """
        Virial stress tensor in eV/Å³, defined as σ_αβ = (1/V) ∂E/∂ε_αβ — the
        strain derivative of the energy at zero strain (V = cell volume).

        This default computes it by central finite differences of a
        deformation-gradient perturbation, which is correct for *any* energy
        expression (used as the fallback for terms without a closed-form
        virial). A strain on entry (α, β) maps each Cartesian coordinate
        r → r + δ·r_β·e_α. Because positions are fractional, deforming the
        cell as lattice → lattice·Fᵀ leaves the fractional coordinates
        unchanged, so only the lattice is perturbed. The 3×3 result is
        symmetrised (the true stress is symmetric; FD noise is not).

        Returns a (3, 3) array. Subclasses with a cheap analytic virial
        (Coulomb, Repulsive) override this.
        """
        lattice = np.asarray(lattice, dtype=float)
        volume  = abs(np.linalg.det(lattice))
        sigma   = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                F = np.eye(3)
                F[a, b] += eps
                e_plus  = self.energy(lattice @ F.T, species, positions)
                F[a, b] -= 2.0 * eps
                e_minus = self.energy(lattice @ F.T, species, positions)
                sigma[a, b] = (e_plus - e_minus) / (2.0 * eps) / volume
        return 0.5 * (sigma + sigma.T)

    def _frac_to_cart(
        self,
        lattice:   np.ndarray,  # (3, 3)
        positions: np.ndarray,  # (N, 3) fractional
    ) -> np.ndarray:
        """Convert fractional coordinates to Cartesian coordinates."""
        return positions @ lattice  # (N, 3)

    def _pbc_distances(
        self,
        cart_positions: np.ndarray,  # (N, 3) Cartesian
        lattice:        np.ndarray,  # (3, 3) rows = lattice vectors a_i
        cutoff:         float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full periodic neighbor list: every (i, j, n) with |r_j + n·L - r_i| <= cutoff,
        over *all* lattice images n ∈ Z³, including self-images (i == j, n ≠ 0).
        Only the true self-pair (i == j, n = 0) is excluded.

        Image range. Let b_i be the reciprocal vectors (a_i · b_j = δ_ij, i.e.
        the columns of L⁻¹) and d_i = 1/|b_i| the perpendicular width of the
        cell along b_i. For a fractional separation s = Δs + n with the
        minimum-image part Δs ∈ [-½, ½]³, the projection r·b̂_i = s_i d_i gives
        |r| >= |s_i| d_i. Hence |r| <= r_c requires |Δs_i + n_i| <= r_c/d_i,
        and it suffices to scan
            |n_i| <= n_i^max = floor(r_c/d_i + ½).
        This holds for any (triclinic) cell and any r_c, including r_c larger
        than the cell.

        Cost is O(N² S) with S = Π(2 n_i^max + 1) shifts, vectorised over an
        (N, N, S, 3) array. Fine for fitting-sized frames (≲ a few hundred
        atoms); large-scale MD uses LAMMPS's own neighbor list.

        Returns an *ordered, full* list: both (i, j, n) and (j, i, -n) appear.
        Pair energies must therefore carry a factor ½; the force on i is
        Σ over entries with first index i.

        Returns:
            i_idx:   (M,) indices of atom i
            j_idx:   (M,) indices of atom j
            r_vecs:  (M, 3) r_j + n·L - r_i in Cartesian (Angstrom)
        """
        n = len(cart_positions)
        if n == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.zeros((0, 3)),
            )

        lattice     = np.asarray(lattice, dtype=float)
        inv_lattice = np.linalg.inv(lattice)                # columns = b_i

        # Minimum-image fractional separations Δs_ij ∈ [-½, ½]³   (N, N, 3)
        frac = cart_positions @ inv_lattice
        ds   = frac[None, :, :] - frac[:, None, :]
        ds  -= np.round(ds)

        # Perpendicular widths d_i = 1/|b_i| and image range n_i^max.
        widths = 1.0 / np.linalg.norm(inv_lattice, axis=0)  # (3,)
        n_max  = np.floor(cutoff / widths + 0.5).astype(int)
        rng    = [np.arange(-m, m + 1) for m in n_max]
        shifts = np.stack(np.meshgrid(*rng, indexing="ij"), axis=-1).reshape(-1, 3)  # (S, 3)
        zero   = int(np.flatnonzero(~shifts.any(axis=1))[0])

        # All image separations (N, N, S, 3) in Cartesian.
        rv   = (ds[:, :, None, :] + shifts[None, None, :, :]) @ lattice
        dist = np.linalg.norm(rv, axis=-1)                  # (N, N, S)
        mask = dist <= cutoff
        idx  = np.arange(n)
        mask[idx, idx, zero] = False                        # drop the self-pair only

        i_idx, j_idx, s_idx = np.nonzero(mask)              # row-major (i, j, shift)
        r_vecs = rv[i_idx, j_idx, s_idx]                    # (M, 3)
        return i_idx, j_idx, r_vecs

    def _neighbors(
        self,
        lattice:   np.ndarray,
        positions: np.ndarray,
        cutoff:    float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cached entry point to the neighbor list for a frame.

        The result depends only on (positions, lattice, cutoff), so we memoize
        it on the byte contents of (positions, lattice) plus cutoff. Subsequent calls
        from any Potential term hit the cache instead of rerunning the N×N PBC
        loop in ``_pbc_distances``.

        The returned arrays are shared — do not mutate them in place.
        """
        key = (positions.tobytes(), lattice.tobytes(), float(cutoff))
        cached = _NEIGHBOR_CACHE.get(key)
        if cached is not None:
            return cached

        cart   = self._frac_to_cart(lattice, positions)
        result = self._pbc_distances(cart, lattice, cutoff)
        _NEIGHBOR_CACHE[key] = result
        return result


# ──────────────────────────────────────────────
# E_c : Coulomb
# ──────────────────────────────────────────────

class Coulomb(Potential):
    """
    Coulomb energy: E_c = ½ k_e Σ_i Σ_(j,n) q_i q_j / r_ij,n   (all images within cutoff)
    Direct truncated summation in eV (k_e = COULOMB_CONSTANT). The truncated
    1/r sum is not convergent in the cutoff; use Ewald for periodic systems.
    """

    def __init__(self, charges: dict[str, float], cutoff: float):
        """
        Args:
            charges: {element: charge} e.g. {"Pb": 1.38, "Ti": 0.99, "O": -0.79}
            cutoff:  cutoff distance in Angstrom
        """
        self.charges = charges
        self.cutoff  = cutoff

    def _charge_vector(self, species: list[str]) -> np.ndarray:
        return np.array([self.charges.get(s, 0.0) for s in species])

    def energy(self, lattice, species, positions) -> float:
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        q     = self._charge_vector(species)
        r     = np.linalg.norm(r_vecs, axis=1)
        return 0.5 * COULOMB_CONSTANT * float(np.sum(q[i_idx] * q[j_idx] / r))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        q  = self._charge_vector(species)
        r  = np.linalg.norm(r_vecs, axis=1)
        # f_i = -dE/dr_i. With r_vec = r_j - r_i, dr/dr_i = -r_vec/r, so the
        # force on i from pair (i,j) is -q_i q_j r_vec/r^3 (per unit pair).
        df = -(COULOMB_CONSTANT * q[i_idx] * q[j_idx] / r**3)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial: σ_αβ = (1/2V) Σ_(i,j,n) f_α r_β over the
        full ordered list (each unordered pair twice), where f is the per-pair
        force on atom i and r = r_j + n·L - r_i. Exact closed form of the
        base-class strain derivative for a central pair potential."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        q    = self._charge_vector(species)
        r    = np.linalg.norm(r_vecs, axis=1)
        df   = -(COULOMB_CONSTANT * q[i_idx] * q[j_idx] / r**3)[:, None] * r_vecs   # force on i per entry
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume


# ──────────────────────────────────────────────
# E_r : Repulsive (Lennard-Jones r^-12)
# ──────────────────────────────────────────────

class Repulsive(Potential):
    """
    Short-range repulsive energy (BVMD form, Liu–Grinberg–Rappe):

        E_r = ½ Σ_i Σ_(j,n) (B_ij / r_ij,n)^12

    B_ij is a length (Å); E_r is in eV. Pairs without a B entry contribute 0.
    """

    def __init__(self, B: dict[str, float], cutoff: float):
        """
        Args:
            B:      {pair: B in Å} e.g. {"Pb-O": 2.17, "Ti-O": 1.28, "O-O": 1.83}
            cutoff: cutoff distance in Angstrom
        """
        self.B      = B
        self.cutoff = cutoff

    def _get_B(self, si: str, sj: str) -> float:
        key1 = f"{si}-{sj}"
        key2 = f"{sj}-{si}"
        return self.B.get(key1, self.B.get(key2, 0.0))

    def _pair_B_table(self, species: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Build (N,) per-atom species code + (K, K) B lookup table for species."""
        uniq     = sorted(set(species))
        code_of  = {s: i for i, s in enumerate(uniq)}
        codes    = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K        = len(uniq)
        B_table  = np.zeros((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                B_table[a, b] = self._get_B(sa, sb)
        return codes, B_table

    def energy(self, lattice, species, positions) -> float:
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        codes, B_table = self._pair_B_table(species)
        r    = np.linalg.norm(r_vecs, axis=1)
        bij  = B_table[codes[i_idx], codes[j_idx]]
        return 0.5 * float(np.sum((bij / r) ** 12))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        codes, B_table = self._pair_B_table(species)
        # Full ordered list: the force on i is the sum over entries (i, j, n).
        # φ(r) = (B/r)^12, φ'(r) = -12 B^12 / r^13. With r_vec = r_j + n·L - r_i,
        # ∂r/∂r_i = -r̂, so f_i = -∂E/∂r_i = φ'(r) r̂ = -12 B^12 r_vec / r^14
        # per entry. Self-image entries (i, i, ±n) cancel pairwise.
        r    = np.linalg.norm(r_vecs, axis=1)
        bij  = B_table[codes[i_idx], codes[j_idx]]
        df   = -(12.0 * bij**12 / r**14)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial: σ_αβ = (1/2V) Σ_(i,j,n) f_α r_β over the
        full ordered list, with f the per-entry force on atom i and
        r = r_j + n·L - r_i."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        codes, B_table = self._pair_B_table(species)
        r    = np.linalg.norm(r_vecs, axis=1)
        bij  = B_table[codes[i_idx], codes[j_idx]]
        df   = -(12.0 * bij**12 / r**14)[:, None] * r_vecs  # force on i per entry
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume


# ──────────────────────────────────────────────
# E_buck : Buckingham (Born-Mayer + dispersion)
# ──────────────────────────────────────────────

class Buckingham(Potential):
    """
    Buckingham pair potential: E = ½ Σ_i Σ_(j,n) A_ij exp(-r / rho_ij) - C_ij / r^6

    The Born-Mayer exponential is the short-range repulsion (softer and more
    physical than the r^-12 of ``Repulsive``); the -C/r^6 term is dispersion
    (set C = 0 for a pure Born-Mayer repulsion). This is the classic pair form
    for ionic oxides and shell-model fits such as PbTiO3.
    """

    def __init__(self, params: dict[str, dict], cutoff: float):
        """
        Args:
            params: {pair: {"A": .., "rho": .., "C": ..}}, e.g.
                    {"O-Ti": {"A": 877.2, "rho": 0.38, "C": 9.0}}
            cutoff: cutoff distance in Angstrom
        """
        self.params = params
        self.cutoff = cutoff

    def _get_pair(self, si: str, sj: str) -> dict | None:
        return self.params.get(f"{si}-{sj}", self.params.get(f"{sj}-{si}", None))

    def _pair_tables(self, species: list[str]):
        """Per-atom codes + (K,K) A, rho, C tables. Unparameterized pairs get
        A=0, rho=1, C=0 so their contribution is exactly zero (no mask)."""
        uniq    = sorted(set(species))
        code_of = {s: i for i, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K       = len(uniq)
        A_tbl   = np.zeros((K, K))
        rho_tbl = np.ones((K, K))
        C_tbl   = np.zeros((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                pair = self._get_pair(sa, sb)
                if pair is not None:
                    A_tbl[a, b]   = pair["A"]
                    rho_tbl[a, b] = pair["rho"]
                    C_tbl[a, b]   = pair["C"]
        return codes, A_tbl, rho_tbl, C_tbl

    def energy(self, lattice, species, positions) -> float:
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        codes, A_tbl, rho_tbl, C_tbl = self._pair_tables(species)
        r    = np.linalg.norm(r_vecs, axis=1)
        A    = A_tbl[codes[i_idx], codes[j_idx]]
        rho  = rho_tbl[codes[i_idx], codes[j_idx]]
        C    = C_tbl[codes[i_idx], codes[j_idx]]
        return 0.5 * float(np.sum(A * np.exp(-r / rho) - C / r**6))

    def _pair_force_df(self, species, i, j, rv, r):
        """Per-pair force on atom i (central): f_i = phi'(r) r_vec / r."""
        codes, A_tbl, rho_tbl, C_tbl = self._pair_tables(species)
        A   = A_tbl[codes[i], codes[j]]
        rho = rho_tbl[codes[i], codes[j]]
        C   = C_tbl[codes[i], codes[j]]
        # phi'(r) = -(A/rho) exp(-r/rho) + 6 C / r^7
        dphi = -(A / rho) * np.exp(-r / rho) + 6.0 * C / r**7
        return (dphi / r)[:, None] * rv

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        # Full ordered list: force on i = sum over entries (i, j, n).
        r    = np.linalg.norm(r_vecs, axis=1)
        df   = self._pair_force_df(species, i_idx, j_idx, r_vecs, r)
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial σ_αβ = (1/2V) Σ_(i,j,n) f_α r_β (full list)."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        r    = np.linalg.norm(r_vecs, axis=1)
        df   = self._pair_force_df(species, i_idx, j_idx, r_vecs, r)
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume


# ──────────────────────────────────────────────
# E_BV : Bond Valence
# ──────────────────────────────────────────────

class BV(Potential):
    """
    Bond valence energy: E_BV = sum_i S_i * (V_i - V0_i)^2
    where V_i = sum_j V_ij is the bond-valence sum. Two standard forms for the
    individual bond valence V_ij are supported via ``form``:

      - "power" (default): V_ij = (r0_ij / r_ij) ^ C_ij      (Brown power law)
      - "exp":             V_ij = exp((r0_ij - r_ij) / b_ij) (Brown-Altermatt)

    The exponential form is the more widely tabulated one (b ≈ 0.37 Å is a
    near-universal value); the power law is retained for backward compatibility.
    """

    def __init__(
        self,
        species_params: dict[str, dict],   # {atom: {V0, S}}
        pair_params:    dict[str, dict],   # {pair: {r0, C, b}}
        cutoff:         float,
        form:           str = "power",     # "power" | "exp"
    ):
        if form not in ("power", "exp"):
            raise ValueError(f"BV form must be 'power' or 'exp', got '{form}'.")
        self.species_params = species_params
        self.pair_params    = pair_params
        self.cutoff         = cutoff
        self.form           = form

    def _get_pair(self, si: str, sj: str) -> dict | None:
        key1 = f"{si}-{sj}"
        key2 = f"{sj}-{si}"
        return self.pair_params.get(key1, self.pair_params.get(key2, None))

    def _bond_valence(self, r: float, r0: float, C: float) -> float:
        return (r0 / r) ** C

    def _pair_tables(self, species: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-atom species codes + (K,K) r0, C and b lookup tables.

        Unparameterized pairs (cation–cation, O–O, …) must give V_ij = 0 and
        dV_ij/dr = 0 in either form, because they would otherwise add to the
        valence of atoms whose S ≠ 0. The r0 sentinel achieves this without a
        mask (C = b = 1):
          power: r0 = 0    → (0/r)^1 = 0,            dV/dr = -1·0/r² = 0
          exp:   r0 = -inf → exp((-inf - r)/1) = 0,   dV/dr = -0/1   = 0
        (The earlier r0 = 0 for both forms gave exp(-r) ≠ 0 for the exp form,
        i.e. spurious valence of ~0.3 v.u. on Pb/Ti and ~0.6 v.u. on O.)
        """
        uniq    = sorted(set(species))
        code_of = {s: i for i, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K       = len(uniq)
        r0_tbl  = np.full((K, K), -np.inf if self.form == "exp" else 0.0)
        C_tbl   = np.ones((K, K))
        b_tbl   = np.ones((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                pair = self._get_pair(sa, sb)
                if pair is not None:
                    r0_tbl[a, b] = pair["r0"]
                    C_tbl[a, b]  = pair["C"]
                    b_tbl[a, b]  = pair.get("b", 0.37)
        return codes, r0_tbl, C_tbl, b_tbl

    def _valence_and_deriv(self, r, r0, C, b):
        """Bond valence V_ij and its radial derivative dV/dr for the active
        ``form``. Vectorised over pairs."""
        if self.form == "exp":
            Vij  = np.exp((r0 - r) / b)
            dVdr = -Vij / b
        else:  # power
            Vij  = (r0 / r) ** C
            dVdr = -C * r0**C / r**(C + 1)
        return Vij, dVdr

    def _species_tables(self, species: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Per-atom V0 and S arrays. Unparameterized atoms get S=0 so they
        contribute zero energy/force, no mask needed."""
        n  = len(species)
        V0 = np.zeros(n)
        S  = np.zeros(n)
        for k, s in enumerate(species):
            sp = self.species_params.get(s)
            if sp is not None:
                V0[k] = sp["V0"]
                S[k]  = sp["S"]
        return V0, S

    def get_valence(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> np.ndarray:
        """Compute bond valence sum V_i for each atom. Returns (N,) array."""
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        V = np.zeros(n)
        if len(i_idx) == 0:
            return V
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        r   = np.linalg.norm(r_vecs, axis=1)
        r0  = r0_tbl[codes[i_idx], codes[j_idx]]
        C   = C_tbl[codes[i_idx], codes[j_idx]]
        b   = b_tbl[codes[i_idx], codes[j_idx]]
        bv, _ = self._valence_and_deriv(r, r0, C, b)
        np.add.at(V, i_idx, bv)
        return V

    def energy(self, lattice, species, positions) -> float:
        V       = self.get_valence(lattice, species, positions)
        V0, S   = self._species_tables(species)
        return float(np.sum(S * (V - V0) ** 2))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        V0_per, S_per        = self._species_tables(species)
        V                    = self.get_valence(lattice, species, positions)

        r    = np.linalg.norm(r_vecs, axis=1)
        r0   = r0_tbl[codes[i_idx], codes[j_idx]]
        C    = C_tbl[codes[i_idx], codes[j_idx]]
        b    = b_tbl[codes[i_idx], codes[j_idx]]
        _, dVdr = self._valence_and_deriv(r, r0, C, b)
        # E_BV = sum_atom S * (V - V0)^2, and r_a moves both V_a (via i-side)
        # AND every V_j where a is in j's neighbours (via j-side). For pair
        # (a, b), f[a] therefore picks up contributions from BOTH the V_a
        # derivative AND the V_b derivative — equal magnitude per-pair but
        # weighted by S_a vs S_b. The original Python-loop implementation
        # omitted the second term.
        dEdr = 2.0 * (
            S_per[i_idx] * (V[i_idx] - V0_per[i_idx])
            + S_per[j_idx] * (V[j_idx] - V0_per[j_idx])
        ) * dVdr
        # f_i = -dE/dr_i. With r_vec = r_j - r_i, dE/dr_i = -dE/dr * r_vec/r,
        # so f_i = +dE/dr * r_vec/r per pair.
        df   = (dEdr / r)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return f

    def energy_and_forces(self, lattice, species, positions):
        """One-pass BV: compute the neighbour list, valence sum, and the
        force scatter together — saves the duplicate `get_valence` call and
        the duplicate (r, r0, C) gather that `energy` + `forces` would do.
        """
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0, np.zeros((n, 3))
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        V0_per, S_per        = self._species_tables(species)

        r   = np.linalg.norm(r_vecs, axis=1)
        r0  = r0_tbl[codes[i_idx], codes[j_idx]]
        C   = C_tbl[codes[i_idx], codes[j_idx]]
        b   = b_tbl[codes[i_idx], codes[j_idx]]
        bv, dVdr = self._valence_and_deriv(r, r0, C, b)
        V   = np.zeros(n)
        np.add.at(V, i_idx, bv)

        e   = float(np.sum(S_per * (V - V0_per) ** 2))

        dEdr = 2.0 * (
            S_per[i_idx] * (V[i_idx] - V0_per[i_idx])
            + S_per[j_idx] * (V[j_idx] - V0_per[j_idx])
        ) * dVdr
        f   = np.zeros((n, 3))
        df  = (dEdr / r)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return e, f


# ──────────────────────────────────────────────
# E_BVV : Bond Valence Vector
# ──────────────────────────────────────────────

class BVV(Potential):
    """
    Bond valence vector energy: E_BVV = sum_i D_i * (|W_i|^2 - W0_i^2)^2
    where W_i = sum_j V_ij * r_hat_ij
    """

    def __init__(
        self,
        species_params: dict[str, dict],   # {atom: {W0, D}}
        pair_params:    dict[str, dict],   # {pair: {r0, C, b}} (same as BV)
        cutoff:         float,
        form:           str = "power",     # "power" | "exp" (same as BV)
    ):
        if form not in ("power", "exp"):
            raise ValueError(f"BVV form must be 'power' or 'exp', got '{form}'.")
        self.species_params = species_params
        self.pair_params    = pair_params
        self.cutoff         = cutoff
        self.form           = form

    def _get_pair(self, si: str, sj: str) -> dict | None:
        key1 = f"{si}-{sj}"
        key2 = f"{sj}-{si}"
        return self.pair_params.get(key1, self.pair_params.get(key2, None))

    def _bond_valence(self, r: float, r0: float, C: float) -> float:
        return (r0 / r) ** C

    def _pair_tables(self, species: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-atom species codes + (K,K) r0, C and b tables (same as BV,
        including the r0 sentinel that zeroes unparameterized pairs)."""
        uniq    = sorted(set(species))
        code_of = {s: i for i, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K       = len(uniq)
        r0_tbl  = np.full((K, K), -np.inf if self.form == "exp" else 0.0)
        C_tbl   = np.ones((K, K))
        b_tbl   = np.ones((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                pair = self._get_pair(sa, sb)
                if pair is not None:
                    r0_tbl[a, b] = pair["r0"]
                    C_tbl[a, b]  = pair["C"]
                    b_tbl[a, b]  = pair.get("b", 0.37)
        return codes, r0_tbl, C_tbl, b_tbl

    def _valence_and_deriv(self, r, r0, C, b):
        """Bond valence V_ij and dV/dr for the active ``form`` (see BV)."""
        if self.form == "exp":
            Vij  = np.exp((r0 - r) / b)
            dVdr = -Vij / b
        else:
            Vij  = (r0 / r) ** C
            dVdr = -C * r0**C / r**(C + 1)
        return Vij, dVdr

    def _species_tables(self, species: list[str]) -> tuple[np.ndarray, np.ndarray]:
        n  = len(species)
        W0 = np.zeros(n)
        D  = np.zeros(n)
        for k, s in enumerate(species):
            sp = self.species_params.get(s)
            if sp is not None:
                W0[k] = sp["W0"]
                D[k]  = sp["D"]
        return W0, D

    def get_bvv(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> np.ndarray:
        """Compute bond valence vector sum W_i for each atom. Returns (N, 3) array."""
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        W = np.zeros((n, 3))
        if len(i_idx) == 0:
            return W
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        r       = np.linalg.norm(r_vecs, axis=1)
        r0      = r0_tbl[codes[i_idx], codes[j_idx]]
        C       = C_tbl[codes[i_idx], codes[j_idx]]
        b       = b_tbl[codes[i_idx], codes[j_idx]]
        Vij, _  = self._valence_and_deriv(r, r0, C, b)
        contrib = (Vij / r)[:, None] * r_vecs       # Vij * r_hat
        np.add.at(W, i_idx, contrib)
        return W

    def energy(self, lattice, species, positions) -> float:
        W       = self.get_bvv(lattice, species, positions)
        W0, D   = self._species_tables(species)
        W2      = np.einsum("ij,ij->i", W, W)
        return float(np.sum(D * (W2 - W0**2) ** 2))

    def forces(self, lattice, species, positions) -> np.ndarray:
        """
        Per-pair force from E_BVV = sum_i D_i (|W_i|^2 - W0_i^2)^2.

        Each pair (a, b) contributes to f[a] via TWO chain-rule branches:
          - V_a's b-term: r_a appears in V_a's b-bond.
          - V_b's a-term: r_a appears in V_b's a-bond (Newton 3rd-law analogue).

        With ∂(V r̂)/∂r_a = (dV/dr - V/r) r̂⊗r̂ - (V/r) I (3×3 tensor; the
        previous implementation collapsed this to a 3-vector and missed both
        the V/r perpendicular term and the b-side contribution).

        Per pair (a, b), contribution to f[a]:
          +4 (dV/dr - V/r) (r̂·[D_a c_a W_a - D_b c_b W_b]) r̂
          +4 (V/r)         (D_a c_a W_a - D_b c_b W_b)
        where c_x = (|W_x|^2 - W0_x^2). Sign is set so f = -∂E/∂r.
        """
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        W0_per, D_per        = self._species_tables(species)
        W                    = self.get_bvv(lattice, species, positions)

        r     = np.linalg.norm(r_vecs, axis=1)
        r_hat = r_vecs / r[:, None]
        r0    = r0_tbl[codes[i_idx], codes[j_idx]]
        C     = C_tbl[codes[i_idx], codes[j_idx]]
        b     = b_tbl[codes[i_idx], codes[j_idx]]
        Vij, dVdr = self._valence_and_deriv(r, r0, C, b)

        W2  = np.einsum("ij,ij->i", W, W)
        W02 = W0_per**2

        c_i = D_per[i_idx] * (W2[i_idx] - W02[i_idx])     # (M,)
        c_j = D_per[j_idx] * (W2[j_idx] - W02[j_idx])     # (M,)
        # Effective per-pair vector v = c_i * W_i - c_j * W_j  (M, 3)
        v = c_i[:, None] * W[i_idx] - c_j[:, None] * W[j_idx]

        v_dot_rhat = np.einsum("mi,mi->m", v, r_hat)      # (M,)
        radial_coeff = 4.0 * (dVdr - Vij / r) * v_dot_rhat  # (M,)
        lateral_coeff = 4.0 * Vij / r                       # (M,)

        # f = -∂E/∂r_a. After collecting both V_a- and V_b-side chain-rule
        # contributions the bracket simplifies to +4(dV/dr - V/r)(v·r̂)r̂ + 4(V/r)v
        # — see derivation in the docstring above.
        df = radial_coeff[:, None] * r_hat + lateral_coeff[:, None] * v
        np.add.at(f, i_idx, df)
        return f

    def energy_and_forces(self, lattice, species, positions):
        """One-pass BVV — shares W, (r, r_hat, r0, C, Vij, dVdr) between
        the energy and force kernels (eliminates a duplicate `get_bvv` call
        plus the duplicate per-pair gathers).
        """
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0, np.zeros((n, 3))
        codes, r0_tbl, C_tbl, b_tbl = self._pair_tables(species)
        W0_per, D_per        = self._species_tables(species)

        r     = np.linalg.norm(r_vecs, axis=1)
        r_hat = r_vecs / r[:, None]
        r0    = r0_tbl[codes[i_idx], codes[j_idx]]
        C     = C_tbl[codes[i_idx], codes[j_idx]]
        b     = b_tbl[codes[i_idx], codes[j_idx]]
        Vij, dVdr = self._valence_and_deriv(r, r0, C, b)

        W   = np.zeros((n, 3))
        np.add.at(W, i_idx, (Vij / r)[:, None] * r_vecs)

        W2   = np.einsum("ij,ij->i", W, W)
        W02  = W0_per**2
        e    = float(np.sum(D_per * (W2 - W02) ** 2))

        c_i = D_per[i_idx] * (W2[i_idx] - W02[i_idx])
        c_j = D_per[j_idx] * (W2[j_idx] - W02[j_idx])
        v   = c_i[:, None] * W[i_idx] - c_j[:, None] * W[j_idx]

        v_dot_rhat   = np.einsum("mi,mi->m", v, r_hat)
        radial_coeff = 4.0 * (dVdr - Vij / r) * v_dot_rhat
        lateral_coeff = 4.0 * Vij / r

        f  = np.zeros((n, 3))
        df = radial_coeff[:, None] * r_hat + lateral_coeff[:, None] * v
        np.add.at(f, i_idx, df)
        return e, f


# ──────────────────────────────────────────────
# E_a : Angle (O-O-O harmonic)
# ──────────────────────────────────────────────

class Angle(Potential):
    """
    Harmonic angle energy for O-O-O angles along octahedral axes.
    E_a = k * sum_i (theta_i - 180)^2
    """

    def __init__(self, k: float, cutoff: float, angle_species: str = "O"):
        """
        Args:
            k:             spring constant in eV/deg^2
            cutoff:        cutoff distance in Angstrom
            angle_species: element to apply angle potential (default: "O")
        """
        self.k             = k
        self.cutoff        = cutoff
        self.angle_species = angle_species

    def _build_triplets(
        self,
        species: list[str],
        i_idx:   np.ndarray,
        j_idx:   np.ndarray,
        r_vecs:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Enumerate every (center i, neighbor a, neighbor b) triple where
        species[i] == angle_species and (a, b) is an unordered pair of i's
        neighbors. Returns five aligned arrays of length T:
            cen, ja, jb : atom indices
            r1, r2      : displacement vectors r_a - r_i, r_b - r_i

        Returns None if no triples exist.
        """
        species_arr = np.asarray(species)
        center_mask = species_arr[i_idx] == self.angle_species
        if not center_mask.any():
            return None
        centers = i_idx[center_mask]
        nbrs_j  = j_idx[center_mask]
        nbrs_r  = r_vecs[center_mask]

        # Group neighbors by their center atom.
        order   = np.argsort(centers, kind="stable")
        centers = centers[order]
        nbrs_j  = nbrs_j[order]
        nbrs_r  = nbrs_r[order]

        unique_c, starts = np.unique(centers, return_index=True)
        counts = np.diff(np.r_[starts, len(centers)])

        cen_parts, a_parts, b_parts = [], [], []
        for c, s, cnt in zip(unique_c, starts, counts):
            if cnt < 2:
                continue
            ia, ib = np.triu_indices(int(cnt), k=1)
            cen_parts.append(np.full(ia.size, c))
            a_parts.append(s + ia)
            b_parts.append(s + ib)

        if not cen_parts:
            return None

        cen   = np.concatenate(cen_parts)
        a_abs = np.concatenate(a_parts)
        b_abs = np.concatenate(b_parts)
        return cen, nbrs_j[a_abs], nbrs_j[b_abs], nbrs_r[a_abs], nbrs_r[b_abs]

    def energy(self, lattice, species, positions) -> float:
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        triples = self._build_triplets(species, i_idx, j_idx, r_vecs)
        if triples is None:
            return 0.0
        _, _, _, r1, r2 = triples
        r1n   = np.linalg.norm(r1, axis=1)
        r2n   = np.linalg.norm(r2, axis=1)
        cos_t = np.clip(np.einsum("mi,mi->m", r1, r2) / (r1n * r2n), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_t))
        return float(self.k * np.sum((theta - 180.0) ** 2))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        f = np.zeros((n, 3))
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return f
        triples = self._build_triplets(species, i_idx, j_idx, r_vecs)
        if triples is None:
            return f
        cen, ja, jb, r1, r2 = triples

        r1n   = np.linalg.norm(r1, axis=1)
        r2n   = np.linalg.norm(r2, axis=1)
        cos_t = np.clip(np.einsum("mi,mi->m", r1, r2) / (r1n * r2n), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_t))
        dEdt  = 2.0 * self.k * (theta - 180.0)
        sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 1e-10))

        rhat1 = r1 / r1n[:, None]
        rhat2 = r2 / r2n[:, None]
        dtdr1 = np.degrees(-(1.0 / (sin_t * r1n))[:, None] * (rhat2 - cos_t[:, None] * rhat1))
        dtdr2 = np.degrees(-(1.0 / (sin_t * r2n))[:, None] * (rhat1 - cos_t[:, None] * rhat2))

        # f = -dE/dr. With dtdr* representing the gradient of theta wrt the
        # respective bond vector (cartesian), the center atom gets the *opposite*
        # of the sum of bond-gradient contributions; the bond-end atoms get
        # the corresponding negative.
        df_i  =  dEdt[:, None] * (dtdr1 + dtdr2)
        df_ja = -dEdt[:, None] * dtdr1
        df_jb = -dEdt[:, None] * dtdr2
        np.add.at(f, cen, df_i)
        np.add.at(f, ja,  df_ja)
        np.add.at(f, jb,  df_jb)
        return f


# ──────────────────────────────────────────────
# BVFF : Total potential
# ──────────────────────────────────────────────

class BVFF:
    """
    Total BVFF potential: E_tot = E_c + E_r + E_BV + E_BVV + E_a
    Each term can be toggled on/off via the terms list.
    """

    def __init__(self, terms: list[Potential]):
        """
        Args:
            terms: list of active Potential instances
        """
        self.terms = terms

    def energy(self, lattice, species, positions) -> float:
        return sum(t.energy(lattice, species, positions) for t in self.terms)

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        f = np.zeros((n, 3))
        for t in self.terms:
            f += t.forces(lattice, species, positions)
        return f

    def stress(self, lattice, species, positions) -> np.ndarray:
        """Total virial stress (3, 3) in eV/Å³, summed over active terms.
        Cheap terms (Coulomb, Repulsive) use their analytic virial; the rest
        fall back to the finite-difference strain derivative."""
        sigma = np.zeros((3, 3))
        for t in self.terms:
            sigma += t.stress(lattice, species, positions)
        return sigma

    def energy_and_forces(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Sum energy and forces over all active terms in a single pass. Each
        term may override ``energy_and_forces`` to share intermediates
        between its energy and force kernels (notably BV.get_valence is
        called once instead of twice).
        """
        n  = len(species)
        e  = 0.0
        f  = np.zeros((n, 3))
        for t in self.terms:
            te, tf = t.energy_and_forces(lattice, species, positions)
            e += te
            f += tf
        return e, f
