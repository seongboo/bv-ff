from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


# Unit conversion for the virial stress. Internally stress is computed in
# eV/Å³ (σ = (1/V) ∂E/∂ε). VASP / vasprun.xml report stress in kBar, so a
# fit against DFT stress needs this factor. 1 eV/Å³ = 160.21766208 GPa
# = 1602.1766208 kBar.
EV_PER_ANG3_TO_KBAR = 1602.1766208

# Coulomb constant in these units: 1 e²/Å = 14.3996 eV. Every electrostatic
# term (direct Coulomb and Ewald) must carry this factor so that charges in
# elementary-charge units yield energies in eV, consistent with the other
# potential terms.
KE_COULOMB = 14.3996


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
#
# The cache is byte-bounded LRU, not an unbounded dict: during fitting the
# keys are the fixed training frames (bounded), but MD/relaxation through
# BVFFCalculator produces a *new* positions key every step (~100 kB per
# 40-atom step) and an unbounded cache is a memory leak over a long run.
# The default bound is generous enough that a typical fitting dataset keeps
# full cross-iteration reuse; raise it for very large datasets if needed.

class _LRUBytesCache:
    """Content-keyed LRU cache bounded by the total byte size of its values."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._data: OrderedDict = OrderedDict()   # key -> (value, nbytes)
        self._nbytes = 0

    def get(self, key):
        entry = self._data.get(key)
        if entry is None:
            return None
        self._data.move_to_end(key)
        return entry[0]

    def put(self, key, value, nbytes: int) -> None:
        if key in self._data:
            return
        self._data[key] = (value, nbytes)
        self._nbytes += nbytes
        while self._nbytes > self.max_bytes and len(self._data) > 1:
            _, (_, evicted) = self._data.popitem(last=False)
            self._nbytes -= evicted

    def clear(self) -> None:
        self._data.clear()
        self._nbytes = 0


NEIGHBOR_CACHE_MAX_BYTES = 512 * 1024 * 1024   # 512 MB ≈ 4500 40-atom frames

_NEIGHBOR_CACHE = _LRUBytesCache(NEIGHBOR_CACHE_MAX_BYTES)


def clear_neighbor_cache() -> None:
    """Drop all cached neighbor lists. Call if frame arrays are mutated in place."""
    _NEIGHBOR_CACHE.clear()


# ──────────────────────────────────────────────
# Cutoff smoothing
# ──────────────────────────────────────────────
#
# A truncated pair term jumps by φ(r_c) every time a pair crosses the cutoff,
# so MD energy is not conserved (each crossing pumps/leaks φ(r_c) plus a force
# spike). Every short-range term therefore supports a C² polynomial switch
# S(r) that tapers its pair kernel to zero over [r_c − w, r_c]:
#
#   S(x) = 1 − 10x³ + 15x⁴ − 6x⁵,  x = (r − (r_c − w)) / w
#
# S, S', S'' all vanish at r = r_c and S=1, S'=S''=0 at r = r_c − w, so
# energy, forces, and the virial are continuous at the cutoff. The Ewald
# real-space term is exempt: erfc(αr_c) is already ≈ the accuracy target by
# construction of alpha, and tapering it would break the real/reciprocal
# split identity.

def _switch(r: np.ndarray, cutoff: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    """C² switching function S(r) (1 → 0 over [cutoff-width, cutoff]) and dS/dr.
    width <= 0 disables smoothing (S ≡ 1, matching plain truncation)."""
    S    = np.ones_like(r)
    dSdr = np.zeros_like(r)
    if width <= 0.0:
        return S, dSdr
    x    = (r - (cutoff - width)) / width
    mask = (x > 0.0) & (x < 1.0)
    xm   = x[mask]
    S[mask]    = 1.0 + xm**3 * (-10.0 + xm * (15.0 - 6.0 * xm))
    dSdr[mask] = (xm**2 * (-30.0 + xm * (60.0 - 30.0 * xm))) / width
    S[x >= 1.0] = 0.0
    return S, dSdr


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

    def param_grads(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> dict[str, tuple[float, np.ndarray]]:
        """
        Analytic parameter gradients: {vector_key: (∂E/∂θ, ∂F/∂θ)} with the
        force gradient shaped (N, 3), for every parameter of THIS term that
        the fit exposes. Keys use the global parameter-vector naming from
        ``fitting.params_to_vector`` (e.g. "repulsive.O-Ti",
        "BV.species.Ti.V0"), so the least-squares Jacobian can be assembled
        column-by-column without finite differences.

        Default: no analytic gradients (the fitting layer falls back to a
        finite-difference column for any vector key no term claims — used for
        the Coulomb/Ewald charges, which are held fixed in the recommended
        fit_charges=0 setup).
        """
        return {}

    def _pair_key_index(self, species: list[str], keyed: dict):
        """Map species-code pairs to an index into ``list(keyed)`` (the term's
        parameterized pair names), resolving both "A-B" and "B-A" orientations.
        Returns (names, (K, K) int table with -1 = no parameter)."""
        uniq   = sorted(set(species))
        names  = list(keyed)
        idx_of = {k: t for t, k in enumerate(names)}
        K   = len(uniq)
        tbl = -np.ones((K, K), dtype=np.int64)
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                t = idx_of.get(f"{sa}-{sb}", idx_of.get(f"{sb}-{sa}", -1))
                tbl[a, b] = t
        return names, tbl

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
        lattice:        np.ndarray,  # (3, 3)
        cutoff:         float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full periodic neighbour list: every image of every pair within ``cutoff``.

        Built with ``ase.neighborlist.neighbor_list``, which enumerates *all*
        periodic images (not just the nearest) within the cutoff and returns one
        entry per image, in both directed orientations ``(i, j)`` and ``(j, i)``.

        This replaces an earlier minimum-image implementation (``np.round`` on the
        fractional displacement). Minimum image keeps only the single nearest image
        of each pair, which is correct **only** when ``cutoff <= L/2`` for every
        lattice dimension. The PbTiO3 example violates this (cell ~7.76 Å, cutoff
        6.0 Å, L/2 ~3.88 Å): minimum image then silently drops ~half of the real
        in-cutoff interactions (e.g. 1520 of 2894 ordered terms for one frame),
        corrupting every term — most damagingly the bond-valence sum V_i = Σ_j V_ij,
        whose missing bonds make (V_i − V0)² meaningless. Enumerating all images
        fixes this for any cutoff/cell. Self-image interactions (i == i, image ≠ 0)
        are included when present; downstream terms handle them via the directed
        ½-sum (their forces cancel by symmetry, their energy is a lattice constant).

        Returns:
            i_idx:   (M,) indices of atom i
            j_idx:   (M,) indices of atom j
            r_vecs:  (M, 3) displacement vectors r_j - r_i in Cartesian (Angstrom),
                     image-shifted so |r_vecs| <= cutoff
        """
        from ase import Atoms
        from ase.neighborlist import neighbor_list

        n = len(cart_positions)
        if n < 2:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.zeros((0, 3)),
            )

        # Chemical identity is irrelevant for a single scalar cutoff, so use a
        # dummy species; only the geometry (positions, cell, pbc) matters.
        atoms = Atoms(
            numbers   = np.ones(n, dtype=int),
            positions = np.asarray(cart_positions, dtype=float),
            cell      = np.asarray(lattice, dtype=float),
            pbc       = True,
        )
        i_idx, j_idx, r_vecs = neighbor_list("ijD", atoms, float(cutoff))
        return i_idx.astype(np.int64), j_idx.astype(np.int64), np.asarray(r_vecs, dtype=float)

    def _neighbors(
        self,
        lattice:   np.ndarray,
        positions: np.ndarray,
        cutoff:    float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cached entry point to the neighbor list for a frame.

        The result depends only on (positions, lattice, cutoff), so we memoize
        it on the (id(positions), id(lattice), cutoff) triple. Subsequent calls
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
        _NEIGHBOR_CACHE.put(
            key, result, sum(a.nbytes for a in result) + len(key[0]) + len(key[1])
        )
        return result


# ──────────────────────────────────────────────
# E_c : Coulomb
# ──────────────────────────────────────────────

class Coulomb(Potential):
    """
    Coulomb energy: E_c = KE * sum_{i<j} q_i * q_j / r_ij, with KE = 14.3996
    eV·Å/e² so charges in |e| give energies in eV (same constant as Ewald).
    Direct summation (use with Ewald for long-range accuracy).

    ``smooth_width`` > 0 tapers each pair with the C² switch — required for
    MD energy conservation when the direct sum is used without Ewald. Note a
    tapered 1/r is a stability device, not accurate electrostatics: use Ewald
    for physical long-range energies.
    """

    def __init__(self, charges: dict[str, float], cutoff: float,
                 smooth_width: float = 0.0):
        """
        Args:
            charges:      {element: charge} e.g. {"Pb": 1.38, "Ti": 0.99, "O": -0.79}
            cutoff:       cutoff distance in Angstrom
            smooth_width: C² taper width in Å (0 = plain truncation)
        """
        self.charges      = charges
        self.cutoff       = cutoff
        self.smooth_width = smooth_width

    def _charge_vector(self, species: list[str]) -> np.ndarray:
        return np.array([self.charges.get(s, 0.0) for s in species])

    def _pair_phi(self, q, i_idx, j_idx, r):
        """Smoothed pair kernel φ(r)·S(r) and its radial derivative."""
        qq    = KE_COULOMB * q[i_idx] * q[j_idx]
        phi   = qq / r
        dphi  = -qq / r**2
        S, dS = _switch(r, self.cutoff, self.smooth_width)
        return phi * S, dphi * S + phi * dS

    def energy(self, lattice, species, positions) -> float:
        # Sum over the full directed neighbour list (both (i,j) and (j,i)) and
        # halve: each unordered interaction is counted once, and self-image
        # interactions (i == i, image ≠ 0) are counted correctly. This replaces
        # an i<j mask, which silently dropped self-images.
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        q = self._charge_vector(species)
        r = np.linalg.norm(r_vecs, axis=1)
        phi, _ = self._pair_phi(q, i_idx, j_idx, r)
        return 0.5 * float(np.sum(phi))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        q  = self._charge_vector(species)
        r  = np.linalg.norm(r_vecs, axis=1)
        # f_i = -dE/dr_i. With r_vec = r_j - r_i, dr/dr_i = -r_vec/r, so the
        # force on i from directed pair (i,j) is (φ'(r)/r)·r_vec.
        # Scattering to i over the full directed list gives each atom the full
        # force (the (j,i) entry supplies j's force), with no ½ factor.
        _, dphi = self._pair_phi(q, i_idx, j_idx, r)
        df = (dphi / r)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial: σ_αβ = (1/V) Σ_{i<j} f_α r_β, where f is
        the per-pair force on atom i and r the displacement r_j - r_i. Computed
        as ½ over the full directed list (equivalent to the i<j sum for i≠j and
        self-image-safe). Validated against the FD path in the tests."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        q  = self._charge_vector(species)
        r  = np.linalg.norm(r_vecs, axis=1)
        _, dphi = self._pair_phi(q, i_idx, j_idx, r)
        df = (dphi / r)[:, None] * r_vecs   # force on i per directed pair
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume


# ──────────────────────────────────────────────
# E_r : Repulsive (Lennard-Jones r^-12)
# ──────────────────────────────────────────────

class Repulsive(Potential):
    """
    Short-range repulsive energy: E_r = sum_{i<j} B_ij / r_ij^12
    (optionally tapered by the C² cutoff switch, see ``smooth_width``).
    """

    def __init__(self, B: dict[str, float], cutoff: float,
                 smooth_width: float = 0.0):
        """
        Args:
            B:            {pair: B} e.g. {"Pb-O": 2.17, "Ti-O": 1.28, "O-O": 1.83}
            cutoff:       cutoff distance in Angstrom
            smooth_width: C² taper width in Å (0 = plain truncation)
        """
        self.B            = B
        self.cutoff       = cutoff
        self.smooth_width = smooth_width

    def _pair_phi(self, bij, r):
        """Smoothed pair kernel φ(r)·S(r) and its radial derivative."""
        phi   = bij / r**12
        dphi  = -12.0 * bij / r**13
        S, dS = _switch(r, self.cutoff, self.smooth_width)
        return phi * S, dphi * S + phi * dS

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
        r   = np.linalg.norm(r_vecs, axis=1)
        bij = B_table[codes[i_idx], codes[j_idx]]
        phi, _ = self._pair_phi(bij, r)
        # ½ × full directed list (counts each interaction once, self-image-safe).
        return 0.5 * float(np.sum(phi))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        codes, B_table = self._pair_B_table(species)
        r   = np.linalg.norm(r_vecs, axis=1)
        bij = B_table[codes[i_idx], codes[j_idx]]
        # f_i = -dE/dr_i. With r_vec = r_j - r_i, the directed-pair (i,j) force
        # on i is (φ'(r)/r)·r_vec. Scatter to i over the full directed list
        # (the (j,i) entry supplies j's force).
        _, dphi = self._pair_phi(bij, r)
        df = (dphi / r)[:, None] * r_vecs
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial: σ_αβ = (1/V) Σ_{i<j} f_α r_β, with f the
        per-pair force on atom i and r = r_j - r_i. Computed as ½ over the full
        directed list (equivalent for i≠j, self-image-safe)."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        codes, B_table = self._pair_B_table(species)
        r   = np.linalg.norm(r_vecs, axis=1)
        bij = B_table[codes[i_idx], codes[j_idx]]
        _, dphi = self._pair_phi(bij, r)
        df  = (dphi / r)[:, None] * r_vecs   # force on i per directed pair
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume

    def param_grads(self, lattice, species, positions):
        """E and F are linear in each B_p: the gradient is the (tapered)
        unit-B kernel summed over that pair's bonds."""
        n = len(species)
        grads = {f"repulsive.{p}": (0.0, np.zeros((n, 3))) for p in self.B}
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return grads
        uniq    = sorted(set(species))
        code_of = {s: k for k, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64,
                              count=n)
        names, key_tbl = self._pair_key_index(species, self.B)
        bond_key = key_tbl[codes[i_idx], codes[j_idx]]
        r = np.linalg.norm(r_vecs, axis=1)
        for t, name in enumerate(names):
            m = bond_key == t
            if not m.any():
                continue
            phi_u, dphi_u = self._pair_phi(np.ones(int(m.sum())), r[m])
            dE = 0.5 * float(np.sum(phi_u))
            dF = np.zeros((n, 3))
            np.add.at(dF, i_idx[m], (dphi_u / r[m])[:, None] * r_vecs[m])
            grads[f"repulsive.{name}"] = (dE, dF)
        return grads


# ──────────────────────────────────────────────
# E_buck : Buckingham (Born-Mayer + dispersion)
# ──────────────────────────────────────────────

# Inner-region guard for Buckingham (see Buckingham._find_guard). E_CAP sets
# the fallback switch radius (-C/r^6 = -E_CAP) for pairs with no repulsive
# wall. _WARNED de-duplicates the per-pair warnings across the many instances
# a fit constructs (one per loss evaluation).
_BUCK_GUARD_E_CAP = 10.0   # eV
_BUCK_GUARD_WARNED: set[tuple[str, str]] = set()


class Buckingham(Potential):
    """
    Buckingham pair potential: E = sum_{i<j} A_ij exp(-r_ij / rho_ij) - C_ij / r_ij^6

    The Born-Mayer exponential is the short-range repulsion (softer and more
    physical than the r^-12 of ``Repulsive``); the -C/r^6 term is dispersion
    (set C = 0 for a pure Born-Mayer repulsion). This is the classic pair form
    for ionic oxides and shell-model fits such as PbTiO3.

    Two MD-safety features:

    * **Inner guard (always on).** With C > 0, φ(r) → −∞ as r → 0 (the
      well-known Buckingham catastrophe): once two atoms tunnel past the
      finite Born-Mayer wall they fuse irreversibly. Below the wall's
      maximum-repulsion point r_sw (the inflection φ''=0) the potential is
      replaced by its tangent line — a constant-force cap that is C² at r_sw,
      keeps φ bounded, and lets the (exponential) BV wall dominate at short
      range. Pairs whose fitted parameters provide no repulsive wall at all
      (A too small vs C) get a fallback cap at −C/r⁶ = −E_CAP and a loud
      warning: such a fit cannot run stable MD.
    * **Cutoff taper** via ``smooth_width`` (C² switch, energy conservation).
    """

    def __init__(self, params: dict[str, dict], cutoff: float,
                 smooth_width: float = 0.0):
        """
        Args:
            params:       {pair: {"A": .., "rho": .., "C": ..}}, e.g.
                          {"O-Ti": {"A": 877.2, "rho": 0.38, "C": 9.0}}
            cutoff:       cutoff distance in Angstrom
            smooth_width: C² taper width in Å (0 = plain truncation)
        """
        self.params       = params
        self.cutoff       = cutoff
        self.smooth_width = smooth_width
        # Per-pair inner guard (r_sw, phi(r_sw), phi'(r_sw)); None = no guard
        # needed (C = 0 → pure Born-Mayer, bounded and repulsive).
        self._guards = {pair: self._find_guard(pair, p) for pair, p in params.items()}

    @staticmethod
    def _find_guard(pair: str, p: dict) -> tuple[float, float, float] | None:
        """Locate the tangent-line switch point for one pair.

        For a healthy wall the switch is the inflection point of φ between the
        turnover maximum and the well — where the repulsive force is largest —
        found numerically on a dense grid (a guard needs ~1e-3 Å accuracy, not
        a root polish; φ'' ≈ 0 there so the tangent join is effectively C²).
        """
        A, rho, C = float(p["A"]), float(p["rho"]), float(p["C"])
        if C <= 0.0:
            return None
        grid = np.linspace(0.02, 3.0, 3000)
        dphi = -(A / rho) * np.exp(-grid / rho) + 6.0 * C / grid**7 if A > 0.0 \
               else 6.0 * C / grid**7
        k = int(np.argmin(dphi))
        if 0 < k < len(grid) - 1 and dphi[k] < 0.0:
            r_sw = float(grid[k])
        else:
            # No repulsive branch (A = 0, or the wall is submerged by C):
            # cap where the dispersion reaches -E_CAP so φ stays bounded.
            r_sw = float(min((C / _BUCK_GUARD_E_CAP) ** (1.0 / 6.0), 2.0))
            key = (pair, "no-wall")
            if key not in _BUCK_GUARD_WARNED:
                _BUCK_GUARD_WARNED.add(key)
                import warnings
                warnings.warn(
                    f"Buckingham pair '{pair}' has no repulsive wall (A too "
                    f"small against C): the inner guard caps it below "
                    f"r={r_sw:.2f} Å, but this parameterization cannot run "
                    f"stable MD — refit or regularize.",
                    stacklevel=2,
                )
        phi_sw  = A * np.exp(-r_sw / rho) - C / r_sw**6
        dphi_sw = -(A / rho) * np.exp(-r_sw / rho) + 6.0 * C / r_sw**7
        return r_sw, float(phi_sw), float(dphi_sw)

    def _get_pair(self, si: str, sj: str) -> dict | None:
        return self.params.get(f"{si}-{sj}", self.params.get(f"{sj}-{si}", None))

    def _get_guard(self, si: str, sj: str):
        g = self._guards.get(f"{si}-{sj}")
        return g if g is not None else self._guards.get(f"{sj}-{si}")

    def _pair_tables(self, species: list[str]):
        """Per-atom codes + (K,K) A, rho, C tables plus the guard tables
        (r_sw, phi_sw, dphi_sw; r_sw=0 → guard never triggers). Unparameterized
        pairs get A=0, rho=1, C=0 so their contribution is exactly zero."""
        uniq    = sorted(set(species))
        code_of = {s: i for i, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K       = len(uniq)
        A_tbl   = np.zeros((K, K))
        rho_tbl = np.ones((K, K))
        C_tbl   = np.zeros((K, K))
        rsw_tbl = np.zeros((K, K))
        psw_tbl = np.zeros((K, K))
        dsw_tbl = np.zeros((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                pair = self._get_pair(sa, sb)
                if pair is not None:
                    A_tbl[a, b]   = pair["A"]
                    rho_tbl[a, b] = pair["rho"]
                    C_tbl[a, b]   = pair["C"]
                    guard = self._get_guard(sa, sb)
                    if guard is not None:
                        rsw_tbl[a, b], psw_tbl[a, b], dsw_tbl[a, b] = guard
        return codes, A_tbl, rho_tbl, C_tbl, rsw_tbl, psw_tbl, dsw_tbl

    def _pair_phi(self, species, i, j, r):
        """Guarded + smoothed pair kernel: (φ·S, (φ·S)') per directed pair."""
        codes, A_tbl, rho_tbl, C_tbl, rsw_tbl, psw_tbl, dsw_tbl = self._pair_tables(species)
        A    = A_tbl[codes[i], codes[j]]
        rho  = rho_tbl[codes[i], codes[j]]
        C    = C_tbl[codes[i], codes[j]]
        phi  = A * np.exp(-r / rho) - C / r**6
        dphi = -(A / rho) * np.exp(-r / rho) + 6.0 * C / r**7
        # Inner guard: tangent-line continuation below r_sw (constant force).
        rsw  = rsw_tbl[codes[i], codes[j]]
        g    = r < rsw
        if g.any():
            psw, dsw = psw_tbl[codes[i], codes[j]], dsw_tbl[codes[i], codes[j]]
            phi[g]  = psw[g] + dsw[g] * (r[g] - rsw[g])
            dphi[g] = dsw[g]
        S, dS = _switch(r, self.cutoff, self.smooth_width)
        return phi * S, dphi * S + phi * dS

    def energy(self, lattice, species, positions) -> float:
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0
        r = np.linalg.norm(r_vecs, axis=1)
        phi, _ = self._pair_phi(species, i_idx, j_idx, r)
        # ½ × full directed list (counts each interaction once, self-image-safe).
        return 0.5 * float(np.sum(phi))

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        f = np.zeros((n, 3))
        if len(i_idx) == 0:
            return f
        r  = np.linalg.norm(r_vecs, axis=1)
        _, dphi = self._pair_phi(species, i_idx, j_idx, r)
        df = (dphi / r)[:, None] * r_vecs   # force on i per directed pair
        # Scatter to i over the full directed list (the (j,i) entry supplies j).
        np.add.at(f, i_idx, df)
        return f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic pairwise virial σ_αβ = (1/V) Σ_{i<j} f_α r_β, computed as ½
        over the full directed list (equivalent for i≠j, self-image-safe)."""
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        if len(i_idx) == 0:
            return np.zeros((3, 3))
        r  = np.linalg.norm(r_vecs, axis=1)
        _, dphi = self._pair_phi(species, i_idx, j_idx, r)
        df = (dphi / r)[:, None] * r_vecs   # force on i per directed pair
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume

    def param_grads(self, lattice, species, positions):
        """Analytic ∂(E, F)/∂θ for θ ∈ {A, rho, C} of every pair:

            ∂φ/∂A = e^(−r/ρ)          ∂φ'/∂A = −(1/ρ)e^(−r/ρ)
            ∂φ/∂ρ = A e^(−r/ρ)·r/ρ²   ∂φ'/∂ρ = (A/ρ²)e^(−r/ρ)(1 − r/ρ)
            ∂φ/∂C = −1/r⁶             ∂φ'/∂C = 6/r⁷

        each tapered as (∂φ·S, ∂φ'·S + ∂φ·S'). The inner-guard region uses the
        raw formulas (the guard's tangent point r_sw also moves with θ, which
        these ignore) — irrelevant in practice because physical training data
        has no pairs below r_sw ≈ 1.3 Å; the guard exists for MD safety, not
        for the fitted region.
        """
        n = len(species)
        grads = {}
        for p in self.params:
            for attr in ("A", "rho", "C"):
                grads[f"buckingham.{p}.{attr}"] = (0.0, np.zeros((n, 3)))
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return grads
        uniq    = sorted(set(species))
        code_of = {s: k for k, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64,
                              count=n)
        names, key_tbl = self._pair_key_index(species, self.params)
        bond_key = key_tbl[codes[i_idx], codes[j_idx]]
        r      = np.linalg.norm(r_vecs, axis=1)
        S, dS  = _switch(r, self.cutoff, self.smooth_width)
        for t, name in enumerate(names):
            m = bond_key == t
            if not m.any():
                continue
            A, rho = self.params[name]["A"], self.params[name]["rho"]
            rm, Sm, dSm = r[m], S[m], dS[m]
            ex = np.exp(-rm / rho)
            per_theta = {
                "A":   (ex,                      -(1.0 / rho) * ex),
                "rho": (A * ex * rm / rho**2,    (A / rho**2) * ex * (1.0 - rm / rho)),
                "C":   (-1.0 / rm**6,            6.0 / rm**7),
            }
            for attr, (dphi_raw, ddphi_raw) in per_theta.items():
                gphi  = dphi_raw * Sm
                gdphi = ddphi_raw * Sm + dphi_raw * dSm
                dE = 0.5 * float(np.sum(gphi))
                dF = np.zeros((n, 3))
                np.add.at(dF, i_idx[m], (gdphi / rm)[:, None] * r_vecs[m])
                grads[f"buckingham.{name}.{attr}"] = (dE, dF)
        return grads


# ──────────────────────────────────────────────
# Bond-valence machinery shared by BV and BVV
# ──────────────────────────────────────────────

class _BondValenceTerm(Potential):
    """Shared machinery for the BV and BVV terms: the pair-parameter lookup
    tables and the bond-valence form V_ij(r) with its radial derivative.
    Two standard forms are supported via ``form``:

      - "power" (default): V_ij = (r0_ij / r_ij) ^ C_ij      (Brown power law)
      - "exp":             V_ij = exp((r0_ij - r_ij) / b_ij) (Brown-Altermatt)

    The exponential form is the more widely tabulated one (b ≈ 0.37 Å is a
    near-universal value); the power law is retained for backward compatibility.

    ``smooth_width`` > 0 tapers V_ij(r) itself with the C² cutoff switch, so
    the valence sums (and hence energy/forces) are continuous when a bond
    crosses the cutoff — required for MD energy conservation.
    """

    def __init__(
        self,
        pair_params:  dict[str, dict],   # {pair: {r0, C, b}}
        cutoff:       float,
        form:         str = "power",     # "power" | "exp"
        smooth_width: float = 0.0,
    ):
        if form not in ("power", "exp"):
            raise ValueError(
                f"{type(self).__name__} form must be 'power' or 'exp', got '{form}'."
            )
        self.pair_params  = pair_params
        self.cutoff       = cutoff
        self.form         = form
        self.smooth_width = smooth_width

    def _get_pair(self, si: str, sj: str) -> dict | None:
        key1 = f"{si}-{sj}"
        key2 = f"{sj}-{si}"
        return self.pair_params.get(key1, self.pair_params.get(key2, None))

    def _pair_tables(self, species: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Per-atom species codes + (K,K) r0, C, b lookup tables + the
        parameterized-pair mask ``on_tbl`` (1.0 where a BV pair is defined,
        else 0.0).

        The mask is load-bearing for the exp form: no (r0, b) value makes
        exp((r0−r)/b) vanish, so an unparameterized pair left at the r0=0,
        b=1 placeholders contributes a PHANTOM e^(−r) bond. An earlier
        version had no mask and relied on "phantom bonds only land on atoms
        with S=0" — silently false the moment every species carries S>0
        (i.e., every real fit). Found by the LAMMPS export cross-validation,
        whose eam/fs tables only ever contained the parameterized pairs.
        (The power form was immune: (0/r)^C = 0.)
        """
        uniq    = sorted(set(species))
        code_of = {s: i for i, s in enumerate(uniq)}
        codes   = np.fromiter((code_of[s] for s in species), dtype=np.int64, count=len(species))
        K       = len(uniq)
        r0_tbl  = np.zeros((K, K))
        C_tbl   = np.ones((K, K))
        b_tbl   = np.ones((K, K))
        on_tbl  = np.zeros((K, K))
        for a, sa in enumerate(uniq):
            for b, sb in enumerate(uniq):
                pair = self._get_pair(sa, sb)
                if pair is not None:
                    r0_tbl[a, b] = pair["r0"]
                    C_tbl[a, b]  = pair["C"]
                    b_tbl[a, b]  = pair.get("b", 0.37)
                    on_tbl[a, b] = 1.0
        return codes, r0_tbl, C_tbl, b_tbl, on_tbl

    def _valence_and_deriv(self, r, r0, C, b, on=None):
        """Bond valence V_ij and its radial derivative dV/dr for the active
        ``form``, tapered by the C² cutoff switch when smooth_width > 0 and
        zeroed on unparameterized pairs via the ``on`` mask (see
        ``_pair_tables``). Vectorised over pairs."""
        if self.form == "exp":
            Vij  = np.exp((r0 - r) / b)
            dVdr = -Vij / b
        else:  # power
            Vij  = (r0 / r) ** C
            dVdr = -C * r0**C / r**(C + 1)
        S, dSdr = _switch(r, self.cutoff, self.smooth_width)
        Vt, dVt = Vij * S, dVdr * S + Vij * dSdr
        if on is not None:
            Vt, dVt = Vt * on, dVt * on
        return Vt, dVt

    def _pair_theta_names(self) -> tuple[str, ...]:
        """Pair-shape parameters the fit exposes for the active form (r0 is
        fit either way; exp fits b, power fits C — mirrors params_to_vector)."""
        return ("r0", "b") if self.form == "exp" else ("r0", "C")

    def _dV_dtheta(self, which, r, r0, C, b, Vt, dVt):
        """∂V/∂θ and ∂(dV/dr)/∂θ for a pair-shape parameter θ, given the
        (tapered) valence Vt and radial derivative dVt. The taper S(r) is
        θ-independent, so ∂(V·S)/∂θ = (∂V_raw/∂θ)·S — and since each ∂V/∂θ
        below is (analytic function of r, θ)·V, the same factor applies to
        the tapered arrays directly; the radial derivative follows from
        commuting the mixed partial: ∂(dV/dr)/∂θ = d(∂V/∂θ)/dr."""
        if self.form == "exp":
            if which == "r0":
                return Vt / b, dVt / b
            if which == "b":
                fac = (r0 - r) / b**2
                return -Vt * fac, -dVt * fac + Vt / b**2
        else:  # power
            if which == "r0":
                return C * Vt / r0, C * dVt / r0
            if which == "C":
                ln = np.log(r0 / r)
                return Vt * ln, dVt * ln - Vt / r
        raise KeyError(f"unknown bond-valence pair parameter '{which}'")


# ──────────────────────────────────────────────
# E_BV : Bond Valence
# ──────────────────────────────────────────────

class BV(_BondValenceTerm):
    """
    Bond valence energy: E_BV = sum_i S_i * (V_i - V0_i)^2
    where V_i = sum_j V_ij is the bond-valence sum (see _BondValenceTerm for
    the supported V_ij forms).
    """

    def __init__(
        self,
        species_params: dict[str, dict],   # {atom: {V0, S}}
        pair_params:    dict[str, dict],   # {pair: {r0, C, b}}
        cutoff:         float,
        form:           str = "power",     # "power" | "exp"
        smooth_width:   float = 0.0,
    ):
        super().__init__(pair_params, cutoff, form, smooth_width)
        self.species_params = species_params

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
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        r   = np.linalg.norm(r_vecs, axis=1)
        r0  = r0_tbl[codes[i_idx], codes[j_idx]]
        C   = C_tbl[codes[i_idx], codes[j_idx]]
        b   = b_tbl[codes[i_idx], codes[j_idx]]
        on  = on_tbl[codes[i_idx], codes[j_idx]]
        bv, _ = self._valence_and_deriv(r, r0, C, b, on)
        np.add.at(V, i_idx, bv)
        return V

    def energy(self, lattice, species, positions) -> float:
        V       = self.get_valence(lattice, species, positions)
        V0, S   = self._species_tables(species)
        return float(np.sum(S * (V - V0) ** 2))

    def _pair_engine(self, lattice, species, positions):
        """Shared one-pass kernel: energy plus the per-directed-pair force df
        on atom i (with the bond vector list). Used by forces,
        energy_and_forces, and the analytic virial.

        E_BV = sum_atom S * (V - V0)^2, and r_a moves both V_a (via i-side)
        AND every V_j where a is in j's neighbours (via j-side). For pair
        (a, b), df therefore picks up contributions from BOTH the V_a
        derivative AND the V_b derivative — equal magnitude per-pair but
        weighted by S_a vs S_b.
        """
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0, i_idx, r_vecs, np.zeros((0, 3))
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        V0_per, S_per        = self._species_tables(species)

        r   = np.linalg.norm(r_vecs, axis=1)
        r0  = r0_tbl[codes[i_idx], codes[j_idx]]
        C   = C_tbl[codes[i_idx], codes[j_idx]]
        b   = b_tbl[codes[i_idx], codes[j_idx]]
        on  = on_tbl[codes[i_idx], codes[j_idx]]
        bv, dVdr = self._valence_and_deriv(r, r0, C, b, on)
        V   = np.zeros(n)
        np.add.at(V, i_idx, bv)

        e   = float(np.sum(S_per * (V - V0_per) ** 2))

        dEdr = 2.0 * (
            S_per[i_idx] * (V[i_idx] - V0_per[i_idx])
            + S_per[j_idx] * (V[j_idx] - V0_per[j_idx])
        ) * dVdr
        # f_i = -dE/dr_i. With r_vec = r_j - r_i, dE/dr_i = -dE/dr * r_vec/r,
        # so f_i = +dE/dr * r_vec/r per pair.
        df = (dEdr / r)[:, None] * r_vecs
        return e, i_idx, r_vecs, df

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        _, i_idx, _, df = self._pair_engine(lattice, species, positions)
        f = np.zeros((n, 3))
        np.add.at(f, i_idx, df)
        return f

    def energy_and_forces(self, lattice, species, positions):
        """One-pass BV: compute the neighbour list, valence sum, and the
        force scatter together — saves the duplicate `get_valence` call and
        the duplicate (r, r0, C) gather that `energy` + `forces` would do.
        """
        n = len(species)
        e, i_idx, _, df = self._pair_engine(lattice, species, positions)
        f = np.zeros((n, 3))
        np.add.at(f, i_idx, df)
        return e, f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic virial. The energy depends on positions only through the
        bond vectors, and the engine's df is the full per-directed-pair force
        on atom i (both S_i and S_j chain-rule shares), so
        σ_αβ = (1/2V) Σ_directed df_α r_β — same identity as the pair terms."""
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        _, _, r_vecs, df = self._pair_engine(lattice, species, positions)
        if len(df) == 0:
            return np.zeros((3, 3))
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume

    def param_grads(self, lattice, species, positions):
        """Analytic ∂(E, F)/∂θ for every fitted BV parameter.

        With A_i ≡ ∂E/∂V_i = 2 S_i (V_i − V0_i), the per-directed-bond force
        is df_m = (A_i + A_j)·dVdr_m·r̂_m, so each θ-column needs ∂A (from ∂S,
        ∂V0, or ∂V_i) and — for pair-shape parameters — ∂dVdr on that pair's
        bonds (see ``_dV_dtheta``).
        """
        n = len(species)
        grads: dict[str, tuple[float, np.ndarray]] = {}
        for atom in self.species_params:
            grads[f"BV.species.{atom}.V0"] = (0.0, np.zeros((n, 3)))
            grads[f"BV.species.{atom}.S"]  = (0.0, np.zeros((n, 3)))
        for p in self.pair_params:
            for attr in self._pair_theta_names():
                grads[f"BV.pairs.{p}.{attr}"] = (0.0, np.zeros((n, 3)))

        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return grads
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        V0_per, S_per = self._species_tables(species)
        r   = np.linalg.norm(r_vecs, axis=1)
        r0  = r0_tbl[codes[i_idx], codes[j_idx]]
        C   = C_tbl[codes[i_idx], codes[j_idx]]
        b   = b_tbl[codes[i_idx], codes[j_idx]]
        on  = on_tbl[codes[i_idx], codes[j_idx]]
        Vt, dVt = self._valence_and_deriv(r, r0, C, b, on)
        V = np.zeros(n)
        np.add.at(V, i_idx, Vt)
        dV    = V - V0_per
        A_per = 2.0 * S_per * dV
        species_arr = np.asarray(species)

        def force_col(dA: np.ndarray, mask=None, ddV_m=None) -> np.ndarray:
            """Assemble ∂F/∂θ from the ∂A column (all bonds) plus, for pair
            parameters, the (A_i+A_j)·∂dVdr part on the masked bonds."""
            dF = np.zeros((n, 3))
            coeff = (dA[i_idx] + dA[j_idx]) * dVt
            np.add.at(dF, i_idx, (coeff / r)[:, None] * r_vecs)
            if mask is not None:
                cm = (A_per[i_idx[mask]] + A_per[j_idx[mask]]) * ddV_m
                np.add.at(dF, i_idx[mask], (cm / r[mask])[:, None] * r_vecs[mask])
            return dF

        for atom in self.species_params:
            sel = species_arr == atom
            dA  = np.where(sel, -2.0 * S_per, 0.0)
            grads[f"BV.species.{atom}.V0"] = (
                float(-np.sum(A_per[sel])), force_col(dA))
            dA  = np.where(sel, 2.0 * dV, 0.0)
            grads[f"BV.species.{atom}.S"] = (
                float(np.sum(dV[sel] ** 2)), force_col(dA))

        names, key_tbl = self._pair_key_index(species, self.pair_params)
        bond_key = key_tbl[codes[i_idx], codes[j_idx]]
        for t, name in enumerate(names):
            m = bond_key == t
            if not m.any():
                continue
            for attr in self._pair_theta_names():
                dV_m, ddV_m = self._dV_dtheta(attr, r[m], r0[m], C[m], b[m],
                                              Vt[m], dVt[m])
                dVi = np.zeros(n)
                np.add.at(dVi, i_idx[m], dV_m)
                dE = float(np.sum(A_per * dVi))
                dA = 2.0 * S_per * dVi
                grads[f"BV.pairs.{name}.{attr}"] = (
                    dE, force_col(dA, mask=m, ddV_m=ddV_m))
        return grads


# ──────────────────────────────────────────────
# E_BVV : Bond Valence Vector
# ──────────────────────────────────────────────

class BVV(_BondValenceTerm):
    """
    Bond valence vector energy: E_BVV = sum_i D_i * (|W_i|^2 - W0_i^2)^2
    where W_i = sum_j V_ij * r_hat_ij (see _BondValenceTerm for the V_ij forms;
    the pair parameters are independent of BV's).
    """

    def __init__(
        self,
        species_params: dict[str, dict],   # {atom: {W0, D}}
        pair_params:    dict[str, dict],   # {pair: {r0, C, b}} (same shape as BV)
        cutoff:         float,
        form:           str = "power",     # "power" | "exp" (same as BV)
        smooth_width:   float = 0.0,
    ):
        super().__init__(pair_params, cutoff, form, smooth_width)
        self.species_params = species_params

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
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        r       = np.linalg.norm(r_vecs, axis=1)
        r0      = r0_tbl[codes[i_idx], codes[j_idx]]
        C       = C_tbl[codes[i_idx], codes[j_idx]]
        b       = b_tbl[codes[i_idx], codes[j_idx]]
        on      = on_tbl[codes[i_idx], codes[j_idx]]
        Vij, _  = self._valence_and_deriv(r, r0, C, b, on)
        contrib = (Vij / r)[:, None] * r_vecs       # Vij * r_hat
        np.add.at(W, i_idx, contrib)
        return W

    def energy(self, lattice, species, positions) -> float:
        W       = self.get_bvv(lattice, species, positions)
        W0, D   = self._species_tables(species)
        W2      = np.einsum("ij,ij->i", W, W)
        return float(np.sum(D * (W2 - W0**2) ** 2))

    def _pair_engine(self, lattice, species, positions):
        """Shared one-pass kernel for E_BVV = sum_i D_i (|W_i|^2 - W0_i^2)^2:
        energy plus the per-directed-pair force df on atom i.

        Each pair (a, b) contributes to f[a] via TWO chain-rule branches:
          - V_a's b-term: r_a appears in V_a's b-bond.
          - V_b's a-term: r_a appears in V_b's a-bond (Newton 3rd-law analogue).

        With ∂(V r̂)/∂r_a = (dV/dr - V/r) r̂⊗r̂ - (V/r) I (3×3 tensor; an early
        implementation collapsed this to a 3-vector and missed both the V/r
        perpendicular term and the b-side contribution).

        Per pair (a, b), contribution to f[a]:
          +4 (dV/dr - V/r) (r̂·[D_a c_a W_a - D_b c_b W_b]) r̂
          +4 (V/r)         (D_a c_a W_a - D_b c_b W_b)
        where c_x = (|W_x|^2 - W0_x^2). Sign is set so f = -∂E/∂r.
        """
        n = len(species)
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return 0.0, i_idx, r_vecs, np.zeros((0, 3))
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        W0_per, D_per        = self._species_tables(species)

        r     = np.linalg.norm(r_vecs, axis=1)
        r_hat = r_vecs / r[:, None]
        r0    = r0_tbl[codes[i_idx], codes[j_idx]]
        C     = C_tbl[codes[i_idx], codes[j_idx]]
        b     = b_tbl[codes[i_idx], codes[j_idx]]
        on    = on_tbl[codes[i_idx], codes[j_idx]]
        Vij, dVdr = self._valence_and_deriv(r, r0, C, b, on)

        W   = np.zeros((n, 3))
        np.add.at(W, i_idx, (Vij / r)[:, None] * r_vecs)

        W2   = np.einsum("ij,ij->i", W, W)
        W02  = W0_per**2
        e    = float(np.sum(D_per * (W2 - W02) ** 2))

        c_i = D_per[i_idx] * (W2[i_idx] - W02[i_idx])     # (M,)
        c_j = D_per[j_idx] * (W2[j_idx] - W02[j_idx])     # (M,)
        # Effective per-pair vector v = c_i * W_i - c_j * W_j  (M, 3)
        v   = c_i[:, None] * W[i_idx] - c_j[:, None] * W[j_idx]

        v_dot_rhat   = np.einsum("mi,mi->m", v, r_hat)    # (M,)
        radial_coeff = 4.0 * (dVdr - Vij / r) * v_dot_rhat
        lateral_coeff = 4.0 * Vij / r

        df = radial_coeff[:, None] * r_hat + lateral_coeff[:, None] * v
        return e, i_idx, r_vecs, df

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        _, i_idx, _, df = self._pair_engine(lattice, species, positions)
        f = np.zeros((n, 3))
        np.add.at(f, i_idx, df)
        return f

    def energy_and_forces(self, lattice, species, positions):
        """One-pass BVV — shares W, (r, r_hat, r0, C, Vij, dVdr) between
        the energy and force kernels (eliminates a duplicate `get_bvv` call
        plus the duplicate per-pair gathers).
        """
        n = len(species)
        e, i_idx, _, df = self._pair_engine(lattice, species, positions)
        f = np.zeros((n, 3))
        np.add.at(f, i_idx, df)
        return e, f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic virial. E_BVV depends on positions only through the bond
        vectors, and df is the full per-directed-pair force on atom i (both
        W_i- and W_j-side chain-rule shares — radial AND lateral components),
        so σ_αβ = (1/2V) Σ_directed df_α r_β. Under i↔j swap both df and r
        flip sign, so the outer product is swap-invariant and the ½ correctly
        collapses the directed double count."""
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        _, _, r_vecs, df = self._pair_engine(lattice, species, positions)
        if len(df) == 0:
            return np.zeros((3, 3))
        return 0.5 * np.einsum("ma,mb->ab", df, r_vecs) / volume

    def param_grads(self, lattice, species, positions):
        """Analytic ∂(E, F)/∂θ for every fitted BVV parameter.

        With c_x = D_x(|W_x|² − W0_x²) and v_m = c_i W_i − c_j W_j, the force
        is df_m = 4(dV−V/r)(v·r̂)r̂ + 4(V/r)v, so a θ-column needs ∂c (all θ),
        ∂W (pair θ only, via ∂V on that pair's bonds), and ∂V/∂dVdr on the
        masked bonds:

            ∂df = 4[(∂dV − ∂V/r)(v·r̂) + (dV − V/r)(∂v·r̂)]r̂
                  + 4(∂V/r)v + 4(V/r)∂v,
            ∂v  = ∂c_i W_i + c_i ∂W_i − ∂c_j W_j − c_j ∂W_j.
        """
        n = len(species)
        grads: dict[str, tuple[float, np.ndarray]] = {}
        for atom in self.species_params:
            grads[f"BVV.{atom}.W0"] = (0.0, np.zeros((n, 3)))
            grads[f"BVV.{atom}.D"]  = (0.0, np.zeros((n, 3)))
        for p in self.pair_params:
            for attr in self._pair_theta_names():
                grads[f"BVV.pairs.{p}.{attr}"] = (0.0, np.zeros((n, 3)))

        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return grads
        codes, r0_tbl, C_tbl, b_tbl, on_tbl = self._pair_tables(species)
        W0_per, D_per = self._species_tables(species)
        r     = np.linalg.norm(r_vecs, axis=1)
        r_hat = r_vecs / r[:, None]
        r0    = r0_tbl[codes[i_idx], codes[j_idx]]
        C     = C_tbl[codes[i_idx], codes[j_idx]]
        b     = b_tbl[codes[i_idx], codes[j_idx]]
        on    = on_tbl[codes[i_idx], codes[j_idx]]
        Vt, dVt = self._valence_and_deriv(r, r0, C, b, on)
        W = np.zeros((n, 3))
        np.add.at(W, i_idx, (Vt / r)[:, None] * r_vecs)
        W2   = np.einsum("ij,ij->i", W, W)
        diff = W2 - W0_per**2
        c    = D_per * diff
        v    = c[i_idx, None] * W[i_idx] - c[j_idx, None] * W[j_idx]
        v_dot_rhat = np.einsum("mi,mi->m", v, r_hat)
        rad_base   = dVt - Vt / r
        species_arr = np.asarray(species)

        def force_col(dc: np.ndarray, dW: np.ndarray | None = None,
                      mask=None, dV_m=None, ddV_m=None) -> np.ndarray:
            dv = dc[i_idx, None] * W[i_idx] - dc[j_idx, None] * W[j_idx]
            if dW is not None:
                dv += c[i_idx, None] * dW[i_idx] - c[j_idx, None] * dW[j_idx]
            dv_dot_rhat = np.einsum("mi,mi->m", dv, r_hat)
            ddf = (4.0 * rad_base * dv_dot_rhat)[:, None] * r_hat \
                + (4.0 * Vt / r)[:, None] * dv
            if mask is not None:
                ddf[mask] += (4.0 * (ddV_m - dV_m / r[mask])
                              * v_dot_rhat[mask])[:, None] * r_hat[mask]
                ddf[mask] += (4.0 * dV_m / r[mask])[:, None] * v[mask]
            dF = np.zeros((n, 3))
            np.add.at(dF, i_idx, ddf)
            return dF

        for atom in self.species_params:
            sel = species_arr == atom
            dc  = np.where(sel, -2.0 * D_per * W0_per, 0.0)
            grads[f"BVV.{atom}.W0"] = (
                float(-4.0 * np.sum(D_per[sel] * diff[sel] * W0_per[sel])),
                force_col(dc))
            dc  = np.where(sel, diff, 0.0)
            grads[f"BVV.{atom}.D"] = (
                float(np.sum(diff[sel] ** 2)), force_col(dc))

        names, key_tbl = self._pair_key_index(species, self.pair_params)
        bond_key = key_tbl[codes[i_idx], codes[j_idx]]
        for t, name in enumerate(names):
            m = bond_key == t
            if not m.any():
                continue
            for attr in self._pair_theta_names():
                dV_m, ddV_m = self._dV_dtheta(attr, r[m], r0[m], C[m], b[m],
                                              Vt[m], dVt[m])
                dW = np.zeros((n, 3))
                np.add.at(dW, i_idx[m], dV_m[:, None] * r_hat[m])
                W_dot_dW = np.einsum("ij,ij->i", W, dW)
                dE = float(4.0 * np.sum(c * W_dot_dW))
                dc = 2.0 * D_per * W_dot_dW
                grads[f"BVV.pairs.{name}.{attr}"] = (
                    dE, force_col(dc, dW=dW, mask=m, dV_m=dV_m, ddV_m=ddV_m))
        return grads


# ──────────────────────────────────────────────
# E_a : Angle (O-O-O harmonic)
# ──────────────────────────────────────────────

class Angle(Potential):
    """
    Harmonic angle energy for O-O-O angles along octahedral axes.
    E_a = k * sum_i (theta_i - 180)^2
    """

    def __init__(self, k: float, cutoff: float, angle_species: str = "O",
                 smooth_width: float = 0.0):
        """
        Args:
            k:             spring constant in eV/deg^2
            cutoff:        cutoff distance in Angstrom
            angle_species: element to apply angle potential (default: "O")
            smooth_width:  C² taper width in Å (0 = plain truncation). Each
                           triple is weighted by S(|r1|)·S(|r2|) so its energy
                           fades in/out continuously as a leg crosses the
                           cutoff — otherwise every crossing injects a finite
                           k·(θ-180)² jump into an MD trajectory.
        """
        self.k             = k
        self.cutoff        = cutoff
        self.angle_species = angle_species
        self.smooth_width  = smooth_width

    def _build_triplets(
        self,
        species: list[str],
        i_idx:   np.ndarray,
        j_idx:   np.ndarray,
        r_vecs:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Enumerate every (center i, neighbor a, neighbor b) triple where the
        center AND both neighbors are ``angle_species``, (a, b) being an
        unordered pair of i's neighbors. Returns five aligned arrays of
        length T:
            cen, ja, jb : atom indices
            r1, r2      : displacement vectors r_a - r_i, r_b - r_i

        Both legs are species-filtered: this is an O-O-O octahedral-axis term
        (per the class docstring). An earlier version filtered only the
        center, so e.g. a 90° Ti-O-Ti bridge (O center, Ti neighbors) was
        penalized toward 180° — a spurious force on every off-axis cation.

        Returns None if no triples exist.
        """
        species_arr = np.asarray(species)
        pair_mask = (species_arr[i_idx] == self.angle_species) \
                  & (species_arr[j_idx] == self.angle_species)
        if not pair_mask.any():
            return None
        centers = i_idx[pair_mask]
        nbrs_j  = j_idx[pair_mask]
        nbrs_r  = r_vecs[pair_mask]

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

    def _triplet_engine(self, lattice, species, positions):
        """Shared kernel: per-triple energy and the bond-vector gradients.

        With taper weights w = S(|r|), E = k Σ_t (θ_t - 180)² w1 w2, so the
        gradient wrt each bond vector has an angular part (through θ) and a
        radial part (through w):
            ∂E/∂r1 = 2kΔθ w1w2 ∂θ/∂r1 + kΔθ² w1' w2 r̂1   (r2 analogous)
        Returns (e, cen, ja, jb, r1, r2, gE1, gE2) with gE* = ∂E/∂r* per
        triple, or None when no triples exist.
        """
        i_idx, j_idx, r_vecs = self._neighbors(lattice, positions, self.cutoff)
        if len(i_idx) == 0:
            return None
        triples = self._build_triplets(species, i_idx, j_idx, r_vecs)
        if triples is None:
            return None
        cen, ja, jb, r1, r2 = triples

        r1n   = np.linalg.norm(r1, axis=1)
        r2n   = np.linalg.norm(r2, axis=1)
        cos_t = np.clip(np.einsum("mi,mi->m", r1, r2) / (r1n * r2n), -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_t))
        dtheta = theta - 180.0
        w1, dw1 = _switch(r1n, self.cutoff, self.smooth_width)
        w2, dw2 = _switch(r2n, self.cutoff, self.smooth_width)

        e = float(self.k * np.sum(dtheta**2 * w1 * w2))

        dEdt  = 2.0 * self.k * dtheta * w1 * w2
        sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 1e-10))

        rhat1 = r1 / r1n[:, None]
        rhat2 = r2 / r2n[:, None]
        dtdr1 = np.degrees(-(1.0 / (sin_t * r1n))[:, None] * (rhat2 - cos_t[:, None] * rhat1))
        dtdr2 = np.degrees(-(1.0 / (sin_t * r2n))[:, None] * (rhat1 - cos_t[:, None] * rhat2))

        gE1 = dEdt[:, None] * dtdr1 + (self.k * dtheta**2 * dw1 * w2)[:, None] * rhat1
        gE2 = dEdt[:, None] * dtdr2 + (self.k * dtheta**2 * w1 * dw2)[:, None] * rhat2
        return e, cen, ja, jb, r1, r2, gE1, gE2

    def energy(self, lattice, species, positions) -> float:
        eng = self._triplet_engine(lattice, species, positions)
        return 0.0 if eng is None else eng[0]

    def forces(self, lattice, species, positions) -> np.ndarray:
        n = len(species)
        f = np.zeros((n, 3))
        eng = self._triplet_engine(lattice, species, positions)
        if eng is None:
            return f
        _, cen, ja, jb, _, _, gE1, gE2 = eng
        # f = -dE/dr. The bond-end atoms get -∂E/∂r* (r* = r_end - r_center);
        # the center gets the opposite of their sum (momentum conservation —
        # all E-dependence on the center position flows through r1 and r2).
        np.add.at(f, cen, gE1 + gE2)
        np.add.at(f, ja,  -gE1)
        np.add.at(f, jb,  -gE2)
        return f

    def energy_and_forces(self, lattice, species, positions):
        n = len(species)
        eng = self._triplet_engine(lattice, species, positions)
        f = np.zeros((n, 3))
        if eng is None:
            return 0.0, f
        e, cen, ja, jb, _, _, gE1, gE2 = eng
        np.add.at(f, cen, gE1 + gE2)
        np.add.at(f, ja,  -gE1)
        np.add.at(f, jb,  -gE2)
        return e, f

    def stress(self, lattice, species, positions, eps: float = 1e-4) -> np.ndarray:
        """Analytic three-body virial: E depends on positions only through the
        bond vectors r1, r2, which transform affinely under strain, so
        σ_αβ = (1/V) Σ_t [(∂E/∂r1)_α r1_β + (∂E/∂r2)_α r2_β]."""
        volume = abs(np.linalg.det(np.asarray(lattice, dtype=float)))
        eng = self._triplet_engine(lattice, species, positions)
        if eng is None:
            return np.zeros((3, 3))
        _, _, _, _, r1, r2, gE1, gE2 = eng
        sigma = (np.einsum("ma,mb->ab", gE1, r1)
                 + np.einsum("ma,mb->ab", gE2, r2)) / volume
        return 0.5 * (sigma + sigma.T)

    def param_grads(self, lattice, species, positions):
        """E and F are linear in k, so ∂(E, F)/∂k is the unit-k evaluation
        (computed with a k=1 clone — robust to the current k being 0)."""
        unit = Angle(k=1.0, cutoff=self.cutoff,
                     angle_species=self.angle_species,
                     smooth_width=self.smooth_width)
        e1, f1 = unit.energy_and_forces(lattice, species, positions)
        return {"angle.k": (e1, f1)}


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
        Every built-in term (Coulomb, Repulsive, Buckingham, BV, BVV, Angle,
        Ewald) has an analytic virial; the base-class finite-difference strain
        derivative remains only as the fallback for future terms."""
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

    def param_grads(
        self,
        lattice:   np.ndarray,
        species:   list[str],
        positions: np.ndarray,
    ) -> dict[str, tuple[float, np.ndarray]]:
        """Merged analytic parameter gradients over all active terms (keys are
        globally unique per params_to_vector's naming; accumulated defensively
        in case two terms ever claim the same key)."""
        merged: dict[str, tuple[float, np.ndarray]] = {}
        for t in self.terms:
            for key, (de, df) in t.param_grads(lattice, species, positions).items():
                if key in merged:
                    pe, pf = merged[key]
                    merged[key] = (pe + de, pf + df)
                else:
                    merged[key] = (de, df)
        return merged
