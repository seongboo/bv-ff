from __future__ import annotations

import fnmatch
import logging
import time
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing, minimize, least_squares

from parsers.parameters_parser import (
    Parameters, CoulombParams, RepulsiveParams,
    BVParams, BVSpecies, BVPair,
    BVVParams, BVVSpecies, AngleParams,
)
from parsers.dataset import Frame
from .outputs import progress_iter
from .potentials import BVFF, EV_PER_ANG3_TO_KBAR


logger = logging.getLogger("bvff")

# Relative threshold for what counts as a patience-resetting improvement.
# dual_annealing's local-refinement phase routinely finds sub-1e-6 reductions
# in the loss; without a threshold every such micro-improvement resets the
# patience counter and the run never terminates short of maxiter. 1e-5 means
# "ignore improvements smaller than 0.001% of the current best".
_PATIENCE_REL_TOL = 1e-5

# How often to print a per-iteration log line during the SA inner loop.
# Improvement lines (patience-reset events) are always logged; non-improvement
# lines are throttled to every _LOG_EVERY iterations. With ~5 evals/s a
# value of 100 gives ~one line every 20 s — readable without flooding the log.
_LOG_EVERY = 100


# ──────────────────────────────────────────────
# Charge neutrality
# ──────────────────────────────────────────────
#
# Ewald is defined only for a neutral cell (the k = 0 term diverges and the
# energy becomes α-dependent otherwise). Fitting each species' charge
# independently would leave the neutral manifold, so one charge — the
# "dependent" species — is eliminated from the parameter vector:
#
#     q_d = -(Σ_{s≠d} n_s q_s) / n_d
#
# The dependent species is the most abundant one (ties → alphabetical),
# i.e. O for oxides. This needs every training frame to share the same
# composition ratio (checked in ``frames_composition``).

def frames_composition(frames: list[Frame]) -> dict[str, int]:
    """Species counts of the first frame; raises if any frame's composition is
    not proportional to it (neutrality by a single constraint would fail)."""
    from collections import Counter
    ref = Counter(frames[0].species)
    tot = sum(ref.values())
    for fr in frames[1:]:
        c = Counter(fr.species)
        n = sum(c.values())
        if set(c) != set(ref) or any(c[s] * tot != ref[s] * n for s in ref):
            raise ValueError(
                "Training frames have different composition ratios "
                f"({dict(ref)} vs {dict(c)}); charge neutrality cannot be imposed "
                "with a single dependent charge."
            )
    return dict(ref)


def dependent_species(composition: dict[str, int]) -> str:
    """Most abundant species (ties → alphabetical); its charge is eliminated."""
    return sorted(composition, key=lambda s: (-composition[s], s))[0]


def neutralize_charges(charges: dict[str, float], composition: dict[str, int]) -> dict[str, float]:
    """Return a copy of ``charges`` with the dependent species' charge set so
    that Σ_s n_s q_s = 0 for ``composition``. Species absent from the
    composition are left untouched (they do not enter the cell)."""
    d   = dependent_species(composition)
    out = dict(charges)
    rest = sum(n * out.get(s, 0.0) for s, n in composition.items() if s != d)
    out[d] = -rest / composition[d]
    return out


# ──────────────────────────────────────────────
# Fixed (held) parameters
# ──────────────────────────────────────────────

def free_mask(keys: list[str], fixed: list[str] | tuple[str, ...]) -> np.ndarray:
    """Boolean mask over ``keys``: True = fitted, False = matches any glob in
    ``fixed`` (held at its input value)."""
    return np.array([not any(fnmatch.fnmatchcase(k, p) for p in fixed) for k in keys], dtype=bool)


# ──────────────────────────────────────────────
# Parameter vector <-> Parameters conversion
# ──────────────────────────────────────────────

def params_to_vector(
    params: Parameters,
    controls_potentials,
    composition: dict[str, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Flatten Parameters dataclass into a 1D numpy array for optimization.

    If ``composition`` is given and Coulomb is active, the dependent species'
    charge (see ``neutralize_charges``) is omitted from the vector; it is
    reconstructed in ``vector_to_params`` so every trial point is neutral.

    Returns:
        vector: (M,) array of parameter values
        keys:   list of parameter names corresponding to each element
    """
    vector, keys = [], []
    pc = controls_potentials

    if pc.use_coulomb:
        skip = dependent_species(composition) if composition else None
        for atom, q in params.coulomb.charges.items():
            if atom == skip:
                continue
            vector.append(q)
            keys.append(f"coulomb.{atom}")

    if pc.use_repulsive:
        for pair, b in params.repulsive.B.items():
            vector.append(b)
            keys.append(f"repulsive.{pair}")

    if getattr(pc, "use_buckingham", False):
        for pair, bp in params.buckingham.pairs.items():
            vector.append(bp.A)
            keys.append(f"buckingham.{pair}.A")
            vector.append(bp.rho)
            keys.append(f"buckingham.{pair}.rho")
            vector.append(bp.C)
            keys.append(f"buckingham.{pair}.C")

    if pc.use_BV:
        # Bond-valence form decides which pair shape parameter is fit:
        #   power → C (exponent), exp → b (decay length). r0 is fit either way.
        bv_form = getattr(pc, "bv_form", "power")
        for atom, sp in params.BV.species.items():
            vector.append(sp.V0)
            keys.append(f"BV.species.{atom}.V0")
            vector.append(sp.S)
            keys.append(f"BV.species.{atom}.S")
        for pair, p in params.BV.pairs.items():
            vector.append(p.r0)
            keys.append(f"BV.pairs.{pair}.r0")
            if bv_form == "exp":
                vector.append(p.b)
                keys.append(f"BV.pairs.{pair}.b")
            else:
                vector.append(p.C)
                keys.append(f"BV.pairs.{pair}.C")

    if pc.use_BVV:
        for atom, sp in params.BVV.species.items():
            vector.append(sp.W0)
            keys.append(f"BVV.{atom}.W0")
            vector.append(sp.D)
            keys.append(f"BVV.{atom}.D")

    if pc.use_angle:
        vector.append(params.angle.k)
        keys.append("angle.k")

    return np.array(vector), keys


def vector_to_params(
    vector: np.ndarray,
    keys: list[str],
    params: Parameters,
    composition: dict[str, int] | None = None,
) -> Parameters:
    """
    Reconstruct Parameters dataclass from a 1D numpy array.

    Args:
        vector:      (M,) optimized parameter values
        keys:        parameter names from params_to_vector
        params:      original Parameters (used as template for structure)
        composition: if given, the dependent charge is recomputed so the
                     cell is neutral (must match the params_to_vector call)

    Returns:
        Updated Parameters dataclass
    """
    import copy
    p = copy.deepcopy(params)

    for val, key in zip(vector, keys):
        parts = key.split(".")

        if parts[0] == "coulomb":
            p.coulomb.charges[parts[1]] = float(val)

        elif parts[0] == "repulsive":
            p.repulsive.B[parts[1]] = float(val)

        elif parts[0] == "buckingham":
            pair, attr = parts[1], parts[2]
            if attr == "A":
                p.buckingham.pairs[pair].A = float(val)
            elif attr == "rho":
                p.buckingham.pairs[pair].rho = float(val)
            elif attr == "C":
                p.buckingham.pairs[pair].C = float(val)

        elif parts[0] == "BV":
            if parts[1] == "species":
                atom, attr = parts[2], parts[3]
                if attr == "V0":
                    p.BV.species[atom].V0 = float(val)
                elif attr == "S":
                    p.BV.species[atom].S = float(val)
            elif parts[1] == "pairs":
                pair, attr = parts[2], parts[3]
                if attr == "r0":
                    p.BV.pairs[pair].r0 = float(val)
                elif attr == "C":
                    p.BV.pairs[pair].C = float(val)
                elif attr == "b":
                    p.BV.pairs[pair].b = float(val)

        elif parts[0] == "BVV":
            atom, attr = parts[1], parts[2]
            if attr == "W0":
                p.BVV.species[atom].W0 = float(val)
            elif attr == "D":
                p.BVV.species[atom].D = float(val)

        elif parts[0] == "angle":
            p.angle.k = float(val)

    if composition and p.coulomb.charges:
        p.coulomb.charges = neutralize_charges(p.coulomb.charges, composition)

    return p


# ──────────────────────────────────────────────
# Energy reference (profiled out of the loss)
# ──────────────────────────────────────────────
#
# DFT total energies have an arbitrary zero; the model must reproduce energy
# *differences*. With a per-species reference E_ref,f = Σ_s n_{f,s} μ_s the
# per-atom energy residual of frame f is
#
#     e_f(μ) = (ΔE_f − N_f·μ) / n_f,     ΔE_f = E_DFT,f − E_BVFF,f
#
# For fixed potential parameters this is linear in μ, so the μ minimising
# Σ_f e_f² is a weighted linear least-squares problem, solved exactly at every
# loss evaluation (variable projection). Rows are scaled by 1/n_f to match the
# per-atom loss. With a single composition N has rank 1: only Σ_s n_s μ_s is
# determined and lstsq returns the minimum-norm μ; E_ref itself is unique.
#
# The profiled residual is invariant to any reference already contained in
# E_BVFF (it lies in the column space of N), so the loss does not depend on
# params.energy_ref.

def _composition_matrix(frames: list[Frame], species: list[str]) -> np.ndarray:
    """(F, K) matrix of atom counts n_{f,s}."""
    col = {s: k for k, s in enumerate(species)}
    N = np.zeros((len(frames), len(species)))
    for r, fr in enumerate(frames):
        for s in fr.species:
            N[r, col[s]] += 1.0
    return N


def _solve_energy_reference(
    delta_e: np.ndarray, N: np.ndarray, n_atoms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (μ, per-atom residuals) for min_μ Σ_f ((ΔE_f − N_f·μ)/n_f)²."""
    A = N / n_atoms[:, None]
    b = delta_e / n_atoms
    mu, *_ = np.linalg.lstsq(A, b, rcond=None)
    return mu, b - A @ mu


def fit_energy_reference(bvff: BVFF, frames: list[Frame]) -> dict[str, float]:
    """Least-squares μ_s for a BVFF *without* reference (its own energy_ref is
    ignored), such that E_DFT ≈ E_BVFF + Σ n_s μ_s."""
    species = sorted({s for fr in frames for s in fr.species})
    N       = _composition_matrix(frames, species)
    n_atoms = N.sum(axis=1)
    delta_e = np.array([
        fr.energy
        - bvff.energy(fr.lattice, fr.species, fr.positions)
        + bvff.reference_energy(fr.species)
        for fr in frames
    ])
    mu, _ = _solve_energy_reference(delta_e, N, n_atoms)
    return {s: float(m) for s, m in zip(species, mu)}


# ──────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────

def compute_loss(
    bvff:           BVFF,
    frames:         list[Frame],
    w_E:            float,
    w_F:            float,
    w_S:            float,
    has_stress:     bool,
    use_stress:     bool = False,
    progress:       bool = False,
    progress_label: str = "loss-eval",
    scales:         dict[str, float] | None = None,
) -> float:
    """
    Compute weighted RMSE loss:
        Loss = w_E * RMSE_E/σ_E + w_F * RMSE_F/σ_F (+ w_S * RMSE_S/σ_S if stress enabled)

    ``scales`` = {"E": σ_E, "F": σ_F, "S": σ_S} makes each term dimensionless
    (see ``data_scales``); None means σ = 1 (raw RMSEs in eV/atom, eV/Å, eV/Å³).

    RMSE_E is per atom, after removing the least-squares per-species
    reference energy (see ``_solve_energy_reference``): only energy
    differences between frames are fitted, never the arbitrary DFT zero.

    Stress is included only when ``use_stress`` and ``has_stress`` are both
    set: the BVFF virial stress (eV/Å³) is compared against the reference,
    which is converted from VASP's kBar to eV/Å³ following the ASE
    convention (σ_eV/Å³ = -σ_kBar / 1602.18 — kBar→eV/Å³ with VASP's sign
    flip). Computing the virial costs up to ~18 extra energy evaluations per
    frame for terms without a closed-form stress (Ewald, BV, BVV, Angle),
    which is why it is opt-in rather than auto-enabled when data carries it.

    When ``progress`` is True, draw an in-place progress bar over the frame
    loop using ``progress_label``. Each loss evaluation gets its own bar that
    fills 0→100%; on a TTY the bar overwrites itself in place.
    """
    f_errors, s_errors = [], []
    e_model, e_ref = [], []
    include_stress = use_stress and has_stress

    frame_iter = progress_iter(frames, label=progress_label) if progress else frames
    for frame in frame_iter:
        n = len(frame.species)

        # Energy + forces in a single pass — Potential subclasses that
        # override `energy_and_forces` share intermediate arrays between
        # the two kernels (notably BV's get_valence and Ewald's reciprocal
        # structure factors).
        e_bvff, f_bvff = bvff.energy_and_forces(
            frame.lattice, frame.species, frame.positions,
        )
        e_model.append(e_bvff)
        e_ref.append(frame.energy)
        f_errors.append(np.mean((f_bvff - frame.forces) ** 2))

        # Stress RMSE (eV/Å³). Reference VASP stress (kBar) → eV/Å³ via the
        # ASE convention (sign flip + unit factor).
        if include_stress and frame.stress is not None:
            s_bvff = bvff.stress(frame.lattice, frame.species, frame.positions)
            s_ref  = -np.asarray(frame.stress) / EV_PER_ANG3_TO_KBAR
            s_errors.append(np.mean((s_bvff - s_ref) ** 2))

    species = sorted({s for fr in frames for s in fr.species})
    N       = _composition_matrix(frames, species)
    _, e_res = _solve_energy_reference(
        np.asarray(e_ref) - np.asarray(e_model), N, N.sum(axis=1),
    )
    sc     = scales or {"E": 1.0, "F": 1.0, "S": 1.0}
    rmse_E = np.sqrt(np.mean(e_res ** 2))
    rmse_F = np.sqrt(np.mean(f_errors))

    loss = w_E * rmse_E / sc["E"] + w_F * rmse_F / sc["F"]

    if has_stress and s_errors:
        rmse_S = np.sqrt(np.mean(s_errors))
        loss  += w_S * rmse_S / sc["S"]

    return float(loss)


# ──────────────────────────────────────────────
# Data scales and residual vector (least squares)
# ──────────────────────────────────────────────

def data_scales(frames: list[Frame], include_stress: bool = False) -> dict[str, float]:
    """
    RMS magnitude of each reference quantity, used to make loss terms
    dimensionless:
        σ_E = RMS of per-atom DFT energies after removing the least-squares
              per-species reference (i.e. the spread the model must explain),
        σ_F = RMS of DFT force components,
        σ_S = RMS of DFT stress components (eV/Å³), if used.
    A zero scale (e.g. a single frame) falls back to 1.
    """
    species = sorted({s for fr in frames for s in fr.species})
    N = _composition_matrix(frames, species)
    _, e_res = _solve_energy_reference(np.array([fr.energy for fr in frames]), N, N.sum(axis=1))
    sE = float(np.sqrt(np.mean(e_res ** 2)))
    sF = float(np.sqrt(np.mean(np.concatenate([fr.forces.ravel() for fr in frames]) ** 2)))
    sS = 1.0
    if include_stress:
        st = [(-np.asarray(fr.stress) / EV_PER_ANG3_TO_KBAR).ravel() for fr in frames if fr.stress is not None]
        if st:
            sS = float(np.sqrt(np.mean(np.concatenate(st) ** 2)))
    fix = lambda v: v if v > 0 else 1.0
    return {"E": fix(sE), "F": fix(sF), "S": fix(sS)}


_NONFINITE_RESIDUAL = 1e6


def residual_vector(
    bvff:           BVFF,
    frames:         list[Frame],
    w_E:            float,
    w_F:            float,
    w_S:            float,
    scales:         dict[str, float],
    include_stress: bool = False,
) -> np.ndarray:
    """
    Residual vector r with
        ||r||² = w_E·MSE_E/σ_E² + w_F·MSE_F/σ_F² (+ w_S·MSE_S/σ_S²),
    i.e. r_E = √(w_E/N_E)·e/σ_E (e: per-atom energy residual after profiling
    out the per-species reference — variable projection), r_F =
    √(w_F/N_F)·ΔF/σ_F, r_S likewise. Non-finite entries (parameter blow-up,
    e.g. (B/r)^12 at extreme B) are replaced by a large constant so the
    trust-region step is rejected rather than the run aborted.
    """
    e_model, e_ref, f_res, s_res = [], [], [], []
    for fr in frames:
        e, f = bvff.energy_and_forces(fr.lattice, fr.species, fr.positions)
        e_model.append(e); e_ref.append(fr.energy)
        f_res.append((f - fr.forces).ravel())
        if include_stress and fr.stress is not None:
            s = bvff.stress(fr.lattice, fr.species, fr.positions)
            s_res.append((s + np.asarray(fr.stress) / EV_PER_ANG3_TO_KBAR).ravel())
    species = sorted({s for fr in frames for s in fr.species})
    N = _composition_matrix(frames, species)
    _, e_res = _solve_energy_reference(np.asarray(e_ref) - np.asarray(e_model), N, N.sum(axis=1))
    f_res = np.concatenate(f_res)
    parts = [np.sqrt(w_E / e_res.size) * e_res / scales["E"],
             np.sqrt(w_F / f_res.size) * f_res / scales["F"]]
    if s_res:
        s_res = np.concatenate(s_res)
        parts.append(np.sqrt(w_S / s_res.size) * s_res / scales["S"])
    r = np.concatenate(parts)
    return np.nan_to_num(r, nan=_NONFINITE_RESIDUAL, posinf=_NONFINITE_RESIDUAL,
                         neginf=-_NONFINITE_RESIDUAL)


def _fd_jacobian(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray,
                 lo: np.ndarray, hi: np.ndarray, r0: np.ndarray | None = None) -> np.ndarray:
    """Forward-difference Jacobian (steps pointed inward at active bounds)."""
    r0 = fun(x) if r0 is None else r0
    J  = np.empty((r0.size, x.size))
    for k in range(x.size):
        h = 1.5e-8 * max(1.0, abs(x[k]))
        if x[k] + h > hi[k]:
            h = -h
        xk = x.copy(); xk[k] += h
        J[:, k] = (fun(xk) - r0) / h
    return J


def log_fit_diagnostics(
    J: np.ndarray, r: np.ndarray, keys: list[str], x: np.ndarray,
    lo: np.ndarray | None = None, hi: np.ndarray | None = None,
    rcond: float = 1e-8, max_corr_lines: int = 10,
) -> dict:
    """
    Identifiability / uncertainty diagnostics from the residual Jacobian at the
    solution.

    - Singular spectrum and condition number of J.
    - Near-null directions (σ_k < rcond·σ_max): parameter combinations the data
      do not constrain; their dominant components are listed. They are removed
      before forming the covariance, otherwise 1/σ_k² swamps everything and
      every correlation reads ±1.
    - Approximate standard errors σ_θ = √diag(s² V Σ⁻² Vᵀ) over the retained
      subspace, with s² = ||r||²/(M−P).
    - Parameters sitting on a bound (their error bars are not meaningful: the
      constraint, not the data, fixes them).
    - The strongest correlations |ρ| > 0.95 (at most ``max_corr_lines``).
    """
    M, P = J.shape
    _, sv, Vt = np.linalg.svd(J, full_matrices=False)
    cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
    keep = sv > rcond * sv[0]
    logger.info(f"Diagnostics: singular values max {sv[0]:.3e}, min {sv[-1]:.3e}, cond {cond:.3e}")

    null_dirs = []
    for k in np.flatnonzero(~keep):
        comp = sorted(zip(keys, Vt[k]), key=lambda t: -abs(t[1]))[:3]
        null_dirs.append((sv[k], comp))
        logger.info(
            f"  unconstrained direction (σ={sv[k]:.1e}): "
            + ", ".join(f"{n} {w:+.2f}" for n, w in comp)
        )

    s2   = float(r @ r) / max(M - P, 1)
    Vk   = Vt[keep].T
    cov  = s2 * (Vk / sv[keep] ** 2) @ Vk.T
    se   = np.sqrt(np.maximum(np.diag(cov), 0.0))

    at_bound = np.zeros(P, dtype=bool)
    if lo is not None and hi is not None:
        tol = 1e-6 * np.maximum(hi - lo, 1e-12)
        at_bound = (x - lo <= tol) | (hi - x <= tol)
    for k_, v, e, b in zip(keys, x, se, at_bound):
        logger.info(f"  {k_:28s} = {v:+.6f} ± {e:.2e}" + ("   [at bound]" if b else ""))

    # Parameters lying (almost) entirely in the removed null space have ~zero
    # retained variance; their correlations are numerical noise.
    degenerate = se <= 1e-8 * (np.abs(x) + 1e-8)
    d    = np.where(se > 0, se, 1.0)
    corr = cov / np.outer(d, d)
    pairs = sorted(
        ((keys[a], keys[b], corr[a, b]) for a in range(P) for b in range(a + 1, P)
         if abs(corr[a, b]) > 0.95 and not (at_bound[a] or at_bound[b])
         and not (degenerate[a] or degenerate[b])),
        key=lambda t: -abs(t[2]),
    )
    for a, b, c in pairs[:max_corr_lines]:
        logger.info(f"  high correlation: {a} ~ {b}  ρ = {c:+.3f}")
    if len(pairs) > max_corr_lines:
        logger.info(f"  … {len(pairs) - max_corr_lines} more pairs with |ρ| > 0.95")
    return {"singular_values": sv, "cond": cond, "stderr": dict(zip(keys, se)),
            "null_directions": null_dirs, "at_bound": [k for k, b in zip(keys, at_bound) if b],
            "correlated": pairs}


# ──────────────────────────────────────────────
# Bounds
# ──────────────────────────────────────────────

def build_bounds(keys: list[str]) -> list[tuple[float, float]]:
    """
    Build parameter bounds for optimization.
    Bounds are set based on physical constraints.
    """
    bounds = []
    for key in keys:
        parts = key.split(".")

        if parts[0] == "coulomb":
            bounds.append((-5.0, 5.0))       # charge

        elif parts[0] == "repulsive":
            bounds.append((0.5, 4.0))         # B in Å for (B/r)^12

        elif parts[0] == "buckingham":
            if parts[-1] == "A":
                bounds.append((0.0, 1e5))     # A >= 0 (eV)
            elif parts[-1] == "rho":
                bounds.append((0.1, 1.0))     # rho in Angstrom
            elif parts[-1] == "C":
                bounds.append((0.0, 1e3))     # dispersion C >= 0

        elif parts[0] == "BV":
            if parts[-1] == "V0":
                bounds.append((0.1, 10.0))    # V0 > 0
            elif parts[-1] == "S":
                bounds.append((0.0, 10.0))    # S >= 0
            elif parts[-1] == "r0":
                bounds.append((0.5, 3.0))     # r0 in Angstrom
            elif parts[-1] == "C":
                bounds.append((1.0, 20.0))    # C > 0
            elif parts[-1] == "b":
                bounds.append((0.05, 1.5))    # decay length b in Angstrom

        elif parts[0] == "BVV":
            if parts[-1] == "W0":
                bounds.append((0.0, 5.0))     # W0 >= 0
            elif parts[-1] == "D":
                bounds.append((0.0, 10.0))    # D >= 0

        elif parts[0] == "angle":
            bounds.append((0.0, 1.0))         # k >= 0

        else:
            bounds.append((0.0, 10.0))        # fallback

    return bounds


# ──────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────

def fit(
    bvff_builder:      Callable[[Parameters], BVFF],
    params:            Parameters,
    controls_potentials,
    train_frames:      list[Frame],
    w_E:               float,
    w_F:               float,
    w_S:               float,
    has_stress:        bool,
    use_stress:        bool  = False,
    polish:            bool  = True,
    maxiter:           int   = 1000,
    target_loss:       float = 0.0,
    patience:          int   = 0,
    seed:              int   = 42,
    fixed:             list[str] | tuple[str, ...] = ("BV.species.*.V0",),
    optimizer:         str   = "lsq",
    n_starts:          int   = 8,
    start_spread:      float = 0.3,
    max_nfev:          int   = 2000,
    jac:               str   = "2-point",
) -> tuple[Parameters, float]:
    """
    Fit BVFF parameters.

    optimizer = "lsq" (default): multi-start bounded nonlinear least squares
    (scipy ``least_squares``, trust-region reflective) on the σ-normalised
    residual vector (see ``residual_vector``). Start 1 is the input
    parameters; starts 2..n multiply each free parameter by U(1-s, 1+s),
    clipped to bounds. The lowest-cost solution is kept.

    optimizer = "sa": Simulated Annealing (``dual_annealing``) on the scalar
    σ-normalised loss, optionally followed by an L-BFGS-B polish.

    Both optimizers use the same σ-normalised loss, so ``best_loss`` is
    comparable: Σ_k w_k RMSE_k/σ_k with σ from ``data_scales``.

    Args:
        bvff_builder:       function that builds BVFF from Parameters
        params:             initial Parameters (all 1.0 or user-defined)
        controls_potentials: PotentialControls from controls.toml
        train_frames:       list of training frames from vasprun.xml
        w_E, w_F, w_S:      loss weights
        has_stress:         whether stress data is available
        use_stress:         include the virial stress term in the loss
        polish:             run L-BFGS-B refinement after SA (keeps best of both)
        maxiter:            hard cap on dual_annealing outer iterations
        target_loss:        stop once best loss < target_loss (0 disables)
        patience:           stop after this many evaluations w/o improvement (0 disables)
        seed:               random seed for reproducibility
        fixed:              glob patterns of parameter keys held at their input
                            values (default: BV V0 — see controls.toml)
        optimizer:          "lsq" | "sa"
        n_starts:           lsq: number of starts
        start_spread:       lsq: relative spread s of random starts
        max_nfev:           lsq: max residual evaluations per start (excl. Jacobian)
        jac:                lsq: finite-difference scheme "2-point" | "3-point"

    Returns:
        fitted_params: optimized Parameters dataclass
        best_loss:     final loss value
    """
    # Charge neutrality: eliminate the dependent charge from the vector and
    # project the starting charges onto the neutral manifold.
    composition = None
    if controls_potentials.use_coulomb and params.coulomb.charges:
        composition = frames_composition(train_frames)
        neutral = neutralize_charges(params.coulomb.charges, composition)
        d = dependent_species(composition)
        if abs(neutral[d] - params.coulomb.charges.get(d, 0.0)) > 1e-12:
            logger.info(
                f"Charge neutrality: initial q_{d} = {params.coulomb.charges.get(d, 0.0):+.6f} "
                f"→ {neutral[d]:+.6f} (composition {composition})"
            )
        import copy
        params = copy.deepcopy(params)
        params.coulomb.charges = neutral
        logger.info(f"Charge neutrality: q_{d} is dependent (not fitted).")

    x_all, keys_all = params_to_vector(params, controls_potentials, composition)
    mask = free_mask(keys_all, fixed)
    for pat in fixed:
        if not any(fnmatch.fnmatchcase(k, pat) for k in keys_all):
            logger.warning(f"fixed pattern '{pat}' matches no fit parameter")
    held = [k for k, m in zip(keys_all, mask) if not m]
    if held:
        logger.info(f"Held fixed ({len(held)}): " + ", ".join(held))
    x0, keys = x_all[mask], [k for k, m in zip(keys_all, mask) if m]
    bounds   = build_bounds(keys)
    lo, hi   = np.array(bounds, dtype=float).T

    if optimizer not in ("lsq", "sa"):
        raise ValueError(f"optimizer must be 'lsq' or 'sa', got '{optimizer}'.")
    include_stress = bool(use_stress and has_stress)
    scales = data_scales(train_frames, include_stress)
    logger.info(
        f"Fitting {len(keys)} parameters with "
        + ("multi-start least squares" if optimizer == "lsq" else "Simulated Annealing") + " ..."
    )
    logger.info(f"Loss weights: w_E={w_E}, w_F={w_F}, w_S={w_S}")
    logger.info(
        f"Data scales: σ_E={scales['E']*1e3:.3f} meV/atom, σ_F={scales['F']:.4f} eV/Å"
        + (f", σ_S={scales['S']:.3e} eV/Å³" if include_stress else "")
        + "  (loss = Σ w·RMSE/σ)"
    )

    def build_x(x: np.ndarray) -> BVFF:
        return bvff_builder(vector_to_params(x, keys, params, composition))

    def resid(x: np.ndarray) -> np.ndarray:
        return residual_vector(build_x(x), train_frames, w_E, w_F, w_S, scales, include_stress)

    def scalar_loss(x: np.ndarray) -> float:
        return compute_loss(build_x(x), train_frames, w_E, w_F, w_S, has_stress,
                            use_stress=use_stress, scales=scales)

    # Show the starting-point loss before optimization begins so the user
    # can see "fitting is alive" immediately. The progress bar streams per-frame
    # progress; every SA evaluation also gets its own bar (see objective).
    logger.info(f"  Computing initial loss on {len(train_frames)} training frames (this can take a while) ...")
    t_fit_start = time.time()
    initial_loss = compute_loss(
        bvff_builder(params), train_frames, w_E, w_F, w_S, has_stress,
        use_stress=use_stress, progress=True, progress_label="initial-loss", scales=scales,
    )
    logger.info(f"  Initial loss = {initial_loss:.6f} (1 evaluation took {time.time() - t_fit_start:.2f}s)")
    eval_s = time.time() - t_fit_start
    if eval_s > 5.0:
        logger.info(
            f"  Note: one loss evaluation costs ~{eval_s:.1f}s; SA will run many of these. "
            f"Consider raising frame_start / lowering train_ratio / using stride to subsample frames."
        )

    if optimizer == "sa":
        if target_loss > 0:
            logger.info(f"  Early stop: target_loss = {target_loss:.6f}")
        if patience > 0:
            logger.info(f"  Early stop: patience    = {patience} evaluations without improvement")

        state = {
            "iter":          0,
            "best":          float(initial_loss),
            "since_improve": 0,
            "t0":            time.time(),
            "stop_reason":   None,
        }

        def objective(x: np.ndarray) -> float:
            # Label each per-iter progress bar with the upcoming iter number so the
            # user can see which evaluation is in flight.
            next_iter = state["iter"] + 1
            current_params = vector_to_params(x, keys, params, composition)
            bvff           = bvff_builder(current_params)
            # progress=False inside the SA inner loop — per-frame progress was
            # flooding the log and slightly slowing things via per-iter I/O.
            loss           = compute_loss(
                bvff, train_frames, w_E, w_F, w_S, has_stress,
                use_stress=use_stress, progress=False, scales=scales,
            )

            state["iter"] = next_iter
            prev_best          = state["best"]
            if loss < prev_best:
                state["best"] = loss
            # Patience only cares about meaningful improvements — see _PATIENCE_REL_TOL.
            improved = loss < prev_best * (1.0 - _PATIENCE_REL_TOL)
            if improved:
                state["since_improve"] = 0
            else:
                state["since_improve"] += 1

            # Decide whether early-stop conditions are met. dual_annealing only
            # acts on this via the callback, but recording it here means the next
            # callback invocation will short-circuit the run.
            if state["stop_reason"] is None:
                if target_loss > 0 and state["best"] <= target_loss:
                    state["stop_reason"] = f"target_loss {target_loss:.6f} reached (best={state['best']:.6f})"
                elif patience > 0 and state["since_improve"] >= patience:
                    state["stop_reason"] = f"no improvement for {patience} evaluations"

            # Throttle non-improvement lines so the log is readable. Always show
            # patience-resetting improvements and stop events.
            should_log = (
                improved
                or state["stop_reason"] is not None
                or next_iter % _LOG_EVERY == 0
            )
            if should_log:
                elapsed = time.time() - state["t0"]
                mark = "*" if improved else " "
                logger.info(
                    f" {mark}iter {next_iter:5d} | loss={loss:.6f} | best={state['best']:.6f} "
                    f"| no-improve {state['since_improve']:4d} | elapsed {elapsed:6.1f}s"
                )

            return loss

        def early_stop_cb(x, f, context):
            # Returning True asks dual_annealing to terminate. We rely on the flag
            # set inside objective() so the trigger semantics are evaluation-based.
            return state["stop_reason"] is not None

        result = dual_annealing(
            objective,
            bounds   = bounds,
            x0       = x0,
            maxiter  = maxiter,
            seed     = seed,
            callback = early_stop_cb,
        )

        best_x    = result.x
        best_loss = float(result.fun)

        if state["stop_reason"]:
            logger.info(f"Early-stopped: {state['stop_reason']}")
        logger.info(
            f"Global stage done in {time.time() - t_fit_start:.1f}s "
            f"| {state['iter']} evaluations | best loss = {best_loss:.6f}"
        )

        # Optional local polish: gradient-based (numerical-gradient) L-BFGS-B from
        # the global best, within the same bounds. Keep it only if it improves.
        if polish:
            logger.info("Polishing best point with L-BFGS-B ...")
            t_polish = time.time()
            polish_res = minimize(
                objective, best_x, method="L-BFGS-B", bounds=bounds,
            )
            polish_loss = float(polish_res.fun)
            if polish_loss < best_loss:
                logger.info(
                    f"  Polish improved loss {best_loss:.6f} → {polish_loss:.6f} "
                    f"({time.time() - t_polish:.1f}s)"
                )
                best_x, best_loss = polish_res.x, polish_loss
            else:
                logger.info(
                    f"  Polish did not improve (kept {best_loss:.6f}, "
                    f"{time.time() - t_polish:.1f}s)"
                )

    else:
        rng    = np.random.default_rng(seed)
        starts = [x0.copy()] + [
            np.clip(x0 * rng.uniform(1.0 - start_spread, 1.0 + start_spread, size=x0.size), lo, hi)
            for _ in range(max(n_starts, 1) - 1)
        ]
        logger.info(
            f"  {len(starts)} starts (spread ±{start_spread:.0%}), max_nfev={max_nfev}, jac={jac}"
        )
        results = []
        for k, xs in enumerate(starts, 1):
            t_s = time.time()
            res = least_squares(
                resid, xs, bounds=(lo, hi), method="trf", x_scale="jac", jac=jac,
                ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev,
            )
            L = scalar_loss(res.x)
            results.append((res.cost, L, res))
            logger.info(
                f"  start {k:2d}/{len(starts)}: loss={L:.6f} (½‖r‖²={res.cost:.4e}) "
                f"nfev={res.nfev} njev={res.njev} status={res.status} | {time.time() - t_s:.1f}s"
            )
        results.sort(key=lambda t: t[0])
        best_cost, best_loss, best_res = results[0]
        best_x = best_res.x
        near = [t for t in results if t[0] <= best_cost * (1 + 1e-6) + 1e-14]
        if len(near) > 1:
            spread = np.max(np.abs(np.array([t[2].x for t in near]) - best_x), axis=0)
            logger.info(
                f"  {len(near)}/{len(results)} starts reached the best cost; "
                f"max parameter spread among them = {spread.max():.2e}"
            )
        else:
            logger.info(f"  best cost reached by 1/{len(results)} starts")
        logger.info(
            f"Fitting stage done in {time.time() - t_fit_start:.1f}s | best loss = {best_loss:.6f}"
        )

    # Identifiability / uncertainty at the solution (P+1 residual evaluations).
    r_best = resid(best_x)
    log_fit_diagnostics(_fd_jacobian(resid, best_x, lo, hi, r_best), r_best, keys, best_x, lo, hi)

    fitted_params = vector_to_params(best_x, keys, params, composition)

    # Reference energy for the fitted model (constant; no effect on forces).
    fitted_params.energy_ref = fit_energy_reference(bvff_builder(fitted_params), train_frames)
    logger.info(
        "Energy reference μ_s (eV/atom): "
        + ", ".join(f"{s}={m:+.6f}" for s, m in fitted_params.energy_ref.items())
    )
    logger.info(
        f"Fitting completed in {time.time() - t_fit_start:.1f}s "
        f"| best loss = {best_loss:.6f}"
    )

    return fitted_params, best_loss
