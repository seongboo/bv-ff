from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing, minimize

from parsers.parameters_parser import (
    Parameters, CoulombParams, RepulsiveParams,
    BVParams, BVSpecies, BVPair,
    BVVParams, BVVSpecies, AngleParams,
    COMMON_ANIONS,
)
from parsers.dataset import Frame
from .outputs import progress_iter
from .potentials import BVFF


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

# Finite penalty returned by compute_loss when a parameter set produces a
# non-finite loss (overflow / division blow-up). Large enough that the
# optimizer always rejects it, but finite so dual_annealing keeps running.
_LOSS_PENALTY = 1e10


# ──────────────────────────────────────────────
# Parameter vector <-> Parameters conversion
# ──────────────────────────────────────────────

def params_to_vector(
    params: Parameters,
    controls_potentials,
    neutral_dep: str | None = None,
    fit_charges: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Flatten Parameters dataclass into a 1D numpy array for optimization.

    When ``neutral_dep`` names a species, that species' charge is *omitted*
    from the vector: it is not a free parameter but is reconstructed from the
    charge-neutrality condition in ``vector_to_params`` (see there). This keeps
    every candidate cell neutral, which Ewald/Coulomb energies require.

    When ``fit_charges`` is False, charges are omitted from the vector entirely
    and stay fixed at their parameters.toml values. This is the recommended
    setting for a bond-valence force field: fitting charges against
    energies+forces alone is under-determined (the Coulomb term is degenerate
    with the repulsion and bond-valence terms), so a free fit collapses the
    charges toward zero — physically wrong for an ionic ferroelectric. The
    canonical Rappe bond-valence potentials hold the charges at fixed formal
    values and fit only the short-range and bond-valence parameters.

    Returns:
        vector: (M,) array of parameter values
        keys:   list of parameter names corresponding to each element
    """
    vector, keys = [], []
    pc = controls_potentials

    if pc.use_coulomb and fit_charges:
        for atom, q in params.coulomb.charges.items():
            if atom == neutral_dep:
                continue          # determined by neutrality, not fit directly
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
        # BVV's own bond-valence pair params (independent of BV). Same form
        # rule as BV: power fits C, exp fits b; r0 is fit either way.
        bvv_form = getattr(pc, "bv_form", "power")
        for pair, p in params.BVV.pairs.items():
            vector.append(p.r0)
            keys.append(f"BVV.pairs.{pair}.r0")
            if bvv_form == "exp":
                vector.append(p.b)
                keys.append(f"BVV.pairs.{pair}.b")
            else:
                vector.append(p.C)
                keys.append(f"BVV.pairs.{pair}.C")

    if pc.use_angle:
        vector.append(params.angle.k)
        keys.append("angle.k")

    return np.array(vector), keys


def vector_to_params(
    vector:      np.ndarray,
    keys:        list[str],
    params:      Parameters,
    neutral_dep: str | None = None,
    counts:      dict[str, int] | None = None,
) -> Parameters:
    """
    Reconstruct Parameters dataclass from a 1D numpy array.

    Args:
        vector:      (M,) optimized parameter values
        keys:        parameter names from params_to_vector
        params:      original Parameters (used as template for structure)
        neutral_dep: species whose charge is fixed by charge neutrality rather
                     than fit. Its charge is set to -Σ_{i≠dep} q_i n_i / n_dep
                     so the cell stays neutral. Requires ``counts``.
        counts:      per-species atom counts in the cell (for the neutrality sum)

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
            if parts[1] == "pairs":
                pair, attr = parts[2], parts[3]
                if attr == "r0":
                    p.BVV.pairs[pair].r0 = float(val)
                elif attr == "C":
                    p.BVV.pairs[pair].C = float(val)
                elif attr == "b":
                    p.BVV.pairs[pair].b = float(val)
            else:
                atom, attr = parts[1], parts[2]
                if attr == "W0":
                    p.BVV.species[atom].W0 = float(val)
                elif attr == "D":
                    p.BVV.species[atom].D = float(val)

        elif parts[0] == "angle":
            p.angle.k = float(val)

    # Charge neutrality: the dependent species' charge is not in the vector;
    # fix it so Σ q_i n_i = 0 over the cell. n_dep > 0 is guaranteed because
    # neutral_dep is chosen as a present species in fit().
    if neutral_dep is not None and counts:
        n_dep = counts.get(neutral_dep, 0)
        if n_dep:
            free_sum = sum(
                p.coulomb.charges[s] * counts.get(s, 0)
                for s in p.coulomb.charges if s != neutral_dep
            )
            p.coulomb.charges[neutral_dep] = -free_sum / n_dep

    return p


# ──────────────────────────────────────────────
# Loss function
# ──────────────────────────────────────────────

def _frame_error(
    bvff:           BVFF,
    frame:          Frame,
    include_stress: bool,
) -> tuple[float, float, float, float | None]:
    """
    Per-frame fit residual summaries for a single frame.

    Pulled out as a module-level function so joblib's process backend can pickle
    it by reference when ``compute_loss`` farms the frame loop out across cores.
    Returns ``(e_bvff_pa, e_ref_pa, f_err, s_err)``:

      - ``e_bvff_pa`` / ``e_ref_pa``: predicted / reference energy *per atom* (eV).
        These are returned separately (not pre-differenced) so ``compute_loss`` can
        remove the optimal constant energy offset across the whole batch — the
        classical BVFF total energy and the DFT total energy live on different
        absolute scales (an ~127 eV gap for PbTiO3 that no parameter can absorb),
        so energies are fit only *up to a constant*.
      - ``f_err``: mean squared force-component error for the frame.
      - ``s_err``: mean squared stress error (``None`` when stress is excluded).
    """
    n = len(frame.species)
    e_bvff, f_bvff = bvff.energy_and_forces(
        frame.lattice, frame.species, frame.positions,
    )
    e_bvff_pa = float(e_bvff / n)
    e_ref_pa  = float(frame.energy / n)
    f_err     = float(np.mean((f_bvff - frame.forces) ** 2))

    s_err: float | None = None
    if include_stress and frame.stress is not None:
        s_bvff = bvff.stress(frame.lattice, frame.species, frame.positions)
        # Frame.stress is already normalized by the readers to eV/Å³, ASE sign
        # (σ = (1/V) ∂E/∂ε) — the same convention as the BVFF virial.
        s_ref  = np.asarray(frame.stress)
        s_err  = float(np.mean((s_bvff - s_ref) ** 2))
    return e_bvff_pa, e_ref_pa, f_err, s_err


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
    n_jobs:         int  = 1,
    norm_E:         float = 1.0,
    norm_F:         float = 1.0,
    norm_S:         float = 1.0,
    subtract_energy_offset: bool = True,
    wE:             np.ndarray | None = None,
    wF:             np.ndarray | None = None,
    wS:             np.ndarray | None = None,
    gid:            np.ndarray | None = None,
) -> float:
    """
    Weighted, scale-normalized RMSE loss:

        Loss = w_E * (RMSE_E / norm_E) + w_F * (RMSE_F / norm_F)
               (+ w_S * RMSE_S if stress enabled)

    Two corrections versus a naive ``w_E*RMSE_E + w_F*RMSE_F``:

    * **Energy offset removed.** Energies are fit *up to a constant*: the optimal
      per-atom shift ``c* = mean(E_bvff/n − E_dft/n)`` is subtracted before
      ``RMSE_E`` (closed-form least-squares offset, no extra parameter). Without
      this, the ~127 eV/cell absolute-scale gap between the classical and DFT
      energies — which no parameter can remove — dominates the loss and the fit
      sacrifices forces chasing it. For a fixed-composition dataset this single
      constant is exactly the per-species reference-energy degree of freedom.
    * **Blocks normalized.** ``RMSE_E`` and ``RMSE_F`` are divided by reference
      scales (``norm_E`` = std of reference per-atom energies, ``norm_F`` = RMS of
      reference force components) so eV and eV/Å are commensurable and ``w_E``/``w_F``
      actually balance the two. With ``norm_E = norm_F = 1`` this reduces to the
      old behavior.

    Stress is included only when ``use_stress`` and ``has_stress`` are both set:
    the BVFF virial stress (eV/Å³) is compared against the reference stress,
    which the readers already normalized to the same convention (eV/Å³, ASE sign).
    Every built-in term has an analytic virial, so the extra cost is roughly
    one force evaluation per frame.

    When ``progress`` is True, draw an in-place progress bar over the frame loop.
    """
    include_stress = use_stress and has_stress

    # Per-frame errors are independent, so with n_jobs != 1 farm the frame loop
    # out to a joblib process pool. loky reuses the same workers across the
    # thousands of loss evaluations a fit performs, so pool start-up is paid once
    # rather than per call. The progress-bar path stays serial: it is used only
    # for the one-off initial-loss eval, where the bar matters more than speed.
    if n_jobs != 1 and not progress:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_frame_error)(bvff, frame, include_stress) for frame in frames
        )
    else:
        # Energy + forces in a single pass per frame — Potential subclasses that
        # override `energy_and_forces` share intermediate arrays between the two
        # kernels (notably BV's get_valence and Ewald's reciprocal structure
        # factors). Stress (eV/Å³) is compared after the ASE kBar→eV/Å³ flip.
        frame_iter = progress_iter(frames, label=progress_label) if progress else frames
        results = [_frame_error(bvff, frame, include_stress) for frame in frame_iter]

    n_fr     = len(results)
    e_bvff   = np.array([r[0] for r in results])
    e_ref    = np.array([r[1] for r in results])
    f_each   = np.array([r[2] for r in results])                       # per-frame mean force MSE
    s_each   = np.array([r[3] if r[3] is not None else np.nan for r in results])

    # Per-frame weights + ref_group ids default to uniform / single group, which
    # reproduces the flat, single-global-offset behavior exactly.
    if wE is None:  wE  = np.ones(n_fr)
    if wF is None:  wF  = np.ones(n_fr)
    if wS is None:  wS  = np.ones(n_fr)
    if gid is None: gid = np.zeros(n_fr, dtype=np.int64)

    de = e_bvff - e_ref
    if subtract_energy_offset and de.size:
        # One weight_E-weighted offset PER ref_group: cross-group reference
        # mismatch is removed while within-group energy differences (the
        # double-well shape) are preserved and fit.
        for g in np.unique(gid):
            m = gid == g
            if wE[m].sum() > 0:
                de[m] -= np.average(de[m], weights=wE[m])
    rmse_E = np.sqrt(np.sum(wE * de ** 2) / np.sum(wE)) if wE.sum() > 0 else 0.0
    rmse_F = np.sqrt(np.sum(wF * f_each) / np.sum(wF)) if wF.sum() > 0 else 0.0

    loss = w_E * (rmse_E / norm_E) + w_F * (rmse_F / norm_F)

    if has_stress:
        sm = np.isfinite(s_each) & (wS > 0)
        if sm.any():
            rmse_S = np.sqrt(np.sum(wS[sm] * s_each[sm]) / np.sum(wS[sm]))
            loss  += w_S * (rmse_S / norm_S)

    # Some parameter regions make a term blow up (e.g. r^-12 at a tiny r, or an
    # overflowing exp), giving NaN/Inf. Return a large finite penalty instead so
    # the optimizer simply rejects the point rather than crashing or stalling.
    if not np.isfinite(loss):
        return _LOSS_PENALTY

    return float(loss)


# ──────────────────────────────────────────────
# Bounds
# ──────────────────────────────────────────────

def build_bounds(keys: list[str]) -> list[tuple[float, float]]:
    """
    Build parameter bounds for optimization.

    Bounds are physical absolute limits per parameter type. They are kept wide
    on purpose: the intended workflow is fitting *from scratch* off the
    auto-generated placeholder parameters (all 1.0), so the global search must
    be free to move charges, exponents, etc. far from those placeholders to
    reach physical values (e.g. an anion charge must be able to go negative).
    Narrowing these around the starting guess would trap a from-scratch fit at
    the placeholders.
    """
    bounds = []
    for key in keys:
        parts = key.split(".")

        if parts[0] == "coulomb":
            # Sign-aware charge bounds: anions confined to [-5, 0], cations to
            # [0, 5]. This is what actually keeps charges physical during the
            # global search (dual_annealing samples the whole box regardless of
            # the starting value), e.g. preventing a cation like Pb from
            # collapsing to ~0. parts[1] is the species symbol (coulomb.<species>).
            if parts[1] in COMMON_ANIONS:
                bounds.append((-5.0, 0.0))   # anion charge <= 0
            else:
                bounds.append((0.0, 5.0))    # cation charge >= 0

        elif parts[0] == "repulsive":
            bounds.append((0.0, 1e4))         # B >= 0

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
            elif parts[-1] == "r0":
                bounds.append((0.5, 3.0))     # r0 in Angstrom (BVV pair)
            elif parts[-1] == "C":
                bounds.append((1.0, 20.0))    # C > 0 (BVV pair)
            elif parts[-1] == "b":
                bounds.append((0.05, 1.5))    # decay length b in Angstrom (BVV pair)

        elif parts[0] == "angle":
            bounds.append((0.0, 1.0))         # k >= 0

        else:
            bounds.append((0.0, 10.0))        # fallback

    return bounds


def bound_saturation_report(
    x:         np.ndarray,
    keys:      list[str],
    bounds:    list[tuple[float, float]],
    rel_tol:   float = 1e-4,
    zero_tol:  float = 1e-10,
    transform: "ParamTransform | None" = None,
) -> list[dict]:
    """
    Flag fitted parameters that sit at/near their box bounds or collapsed to ~0.

    A bound-saturated optimum is an identifiability symptom, not a converged
    physical fit: the optimizer wanted to leave the feasible region (or delete a
    term entirely, e.g. a repulsive B → 1e-32 leaves no short-range wall between
    that pair — an MD-stability landmine).

    Saturation is judged in the space the optimizer actually searched: when
    ``transform`` is given, log-space magnitude parameters are checked in
    log10 — a magnitude legitimately sitting orders of magnitude below its
    linear upper bound (e.g. B = 0.8 with bounds (0, 1e4)) is NOT flagged,
    while a walk to within one decade of the effective-zero floor IS (values
    there are physically negligible). Linear parameters use ``rel_tol`` of the
    bound span; ``|value| <= zero_tol`` is flagged as collapse regardless.
    Reported values/bounds are always physical-space.

    Returns a list of ``{"key", "value", "lower", "upper", "flag", "space"}``
    dicts, ``flag`` ∈ {"at_lower_bound", "at_upper_bound", "collapsed_to_zero"}.
    """
    flags = []
    xs = np.asarray(x, dtype=float)
    for i, (xi, key, (lo, hi)) in enumerate(zip(xs, keys, bounds)):
        is_log = transform is not None and bool(transform.is_log[i])
        if abs(xi) <= zero_tol:
            kind = "collapsed_to_zero"
        elif is_log:
            zlo, zhi = np.log10(transform.lo_eff[i]), np.log10(hi)
            zi   = np.log10(min(max(xi, transform.lo_eff[i]), hi))
            ztol = rel_tol * (zhi - zlo)
            # Lower side: within a decade of the floor = effectively deleted.
            if zi - zlo <= max(ztol, 1.0):
                kind = "at_lower_bound"
            elif zhi - zi <= ztol:
                kind = "at_upper_bound"
            else:
                continue
        else:
            tol = rel_tol * (hi - lo)
            if xi - lo <= tol:
                kind = "at_lower_bound"
            elif hi - xi <= tol:
                kind = "at_upper_bound"
            else:
                continue
        flags.append({
            "key":   key,
            "value": float(xi),
            "lower": float(lo),
            "upper": float(hi),
            "flag":  kind,
            "space": "log10" if is_log else "linear",
        })
    return flags


# ──────────────────────────────────────────────
# Optimizer-space transform (log-space magnitudes)
# ──────────────────────────────────────────────

def _is_log_key(key: str) -> bool:
    """Magnitude-like parameters that are optimized as log10(value): strictly
    non-negative prefactors that legitimately span orders of magnitude and
    whose collapse to zero should be *visible* (a walk to the log lower bound)
    rather than a silent 1e-37. Shape parameters (r0, rho, b, C exponent, V0,
    W0, charges, angle k) stay linear."""
    parts = key.split(".")
    if parts[0] == "repulsive":
        return True                                        # pair prefactor B
    if parts[0] == "buckingham" and parts[-1] in ("A", "C"):
        return True                                        # Born-Mayer A, dispersion C
    if parts[0] == "BV" and parts[1] == "species" and parts[-1] == "S":
        return True                                        # BV stiffness S
    if parts[0] == "BVV" and parts[1] != "pairs" and parts[-1] == "D":
        return True                                        # BVV stiffness D
    return False


class ParamTransform:
    """
    Bijection between the physical parameter space and the optimizer space.

    With ``log_space=True`` (default), magnitude parameters (see
    ``_is_log_key``) are optimized as log10(value): their zero lower bound is
    replaced by a positive floor (1e-8 × max(upper, 1) — small enough that the
    term's contribution is physically negligible there), so "collapse to zero"
    becomes a visible walk to the log lower bound and the finite-difference
    Jacobian is well-conditioned across the orders of magnitude these
    prefactors span. Seed values below the floor (e.g. a deliberate 0.0) are
    clipped up to it. With ``log_space=False`` this is the identity.

    Instances hold only plain arrays, so joblib can ship them to worker
    processes for the multi-start ensemble.
    """

    def __init__(self, keys: list[str], bounds: list[tuple[float, float]],
                 log_space: bool = True):
        self.keys   = list(keys)
        self.lo     = np.array([b[0] for b in bounds], dtype=float)
        self.hi     = np.array([b[1] for b in bounds], dtype=float)
        self.is_log = np.array(
            [log_space and _is_log_key(k) for k in keys], dtype=bool
        )
        floor       = 1e-8 * np.maximum(self.hi, 1.0)
        self.lo_eff = np.where(self.is_log & (self.lo <= 0.0),
                               floor, np.maximum(self.lo, 1e-300))

    def to_opt(self, x: np.ndarray) -> np.ndarray:
        """Physical → optimizer space (log params clipped into [lo_eff, hi])."""
        z = np.array(x, dtype=float)
        m = self.is_log
        z[m] = np.log10(np.clip(z[m], self.lo_eff[m], self.hi[m]))
        return z

    def from_opt(self, z: np.ndarray) -> np.ndarray:
        """Optimizer → physical space."""
        x = np.array(z, dtype=float)
        m = self.is_log
        x[m] = 10.0 ** x[m]
        return x

    def opt_bounds(self) -> list[tuple[float, float]]:
        out = []
        for i in range(self.lo.size):
            if self.is_log[i]:
                out.append((float(np.log10(self.lo_eff[i])),
                            float(np.log10(self.hi[i]))))
            else:
                out.append((float(self.lo[i]), float(self.hi[i])))
        return out


# ──────────────────────────────────────────────
# Single annealing run (one seed) — used directly for a single start and
# fanned out across processes for a parallel multi-start ensemble.
# ──────────────────────────────────────────────

def _single_anneal(
    seed:         int,
    x0:           np.ndarray,
    bounds:       list[tuple[float, float]],
    keys:         list[str],
    params:       Parameters,
    neutral_dep:  str | None,
    counts:       dict[str, int],
    bvff_builder: Callable[[Parameters], BVFF],
    train_frames: list[Frame],
    w_E:          float,
    w_F:          float,
    w_S:          float,
    has_stress:   bool,
    use_stress:   bool,
    maxiter:      int,
    target_loss:  float,
    patience:     int,
    initial_loss: float,
    norm_E:       float = 1.0,
    norm_F:       float = 1.0,
    norm_S:       float = 1.0,
    wE:           np.ndarray | None = None,
    wF:           np.ndarray | None = None,
    wS:           np.ndarray | None = None,
    gid:          np.ndarray | None = None,
    lambda_reg:   float = 0.0,
    x0_seed:      np.ndarray | None = None,
    n_jobs:       int  = 1,
    verbose:      bool = True,
    transform:    "ParamTransform | None" = None,
) -> tuple[np.ndarray, float, int, str | None]:
    """
    Run one ``dual_annealing`` optimization from ``x0`` with the given ``seed``.

    Module-level (not a closure) so joblib's process backend can ship it to a
    worker for the parallel multi-start ensemble. When ``verbose`` is False the
    per-iteration log lines are suppressed — worker-process logging does not
    stream into the main log file, so parallel starts run quietly and report
    only a one-line summary back in ``fit``.

    When ``transform`` is given, ``x0``/``bounds`` and the returned ``best_x``
    live in the OPTIMIZER space (log-space magnitudes); the objective maps back
    to physical space per evaluation. ``x0_seed`` (the Tikhonov anchor) is
    always physical-space.

    Returns ``(best_x, best_loss, n_evals, stop_reason)``.
    """
    state = {
        "iter":          0,
        "best":          float(initial_loss),
        "since_improve": 0,
        "t0":            time.time(),
        "stop_reason":   None,
    }

    def objective(x: np.ndarray) -> float:
        next_iter      = state["iter"] + 1
        x_phys         = transform.from_opt(x) if transform is not None else np.asarray(x)
        current_params = vector_to_params(x_phys, keys, params, neutral_dep, counts)
        bvff           = bvff_builder(current_params)
        loss           = compute_loss(
            bvff, train_frames, w_E, w_F, w_S, has_stress,
            use_stress=use_stress, progress=False, n_jobs=n_jobs,
            norm_E=norm_E, norm_F=norm_F, norm_S=norm_S,
            wE=wE, wF=wF, wS=wS, gid=gid,
        )
        if lambda_reg > 0 and x0_seed is not None:
            scale = np.maximum(np.abs(x0_seed), 1e-8)
            loss += lambda_reg * float(np.mean(((x_phys - x0_seed) / scale) ** 2))

        state["iter"] = next_iter
        prev_best     = state["best"]
        if loss < prev_best:
            state["best"] = loss
        improved = loss < prev_best * (1.0 - _PATIENCE_REL_TOL)
        if improved:
            state["since_improve"] = 0
        else:
            state["since_improve"] += 1

        if state["stop_reason"] is None:
            if target_loss > 0 and state["best"] <= target_loss:
                state["stop_reason"] = f"target_loss {target_loss:.6f} reached (best={state['best']:.6f})"
            elif patience > 0 and state["since_improve"] >= patience:
                state["stop_reason"] = f"no improvement for {patience} evaluations"

        if verbose:
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
        return state["stop_reason"] is not None

    result = dual_annealing(
        objective,
        bounds   = bounds,
        x0       = x0,
        maxiter  = maxiter,
        seed     = seed,
        callback = early_stop_cb,
    )
    return result.x, float(result.fun), state["iter"], state["stop_reason"]


# ──────────────────────────────────────────────
# Parallel finite-difference gradient for the L-BFGS-B polish
# ──────────────────────────────────────────────

def _fd_grad_parallel(
    x:       np.ndarray,
    f0:      float,
    loss_fn: Callable[[np.ndarray], float],
    bounds:  list[tuple[float, float]],
    n_jobs:  int,
    eps:     float = 1e-6,
) -> np.ndarray:
    """
    Forward-difference gradient of ``loss_fn`` at ``x``, perturbations in parallel.

    ``g_i = (loss_fn(x + h_i e_i) - f0) / h_i`` with a per-component step scaled to
    the variable and flipped to a backward step when a forward step would cross
    the upper bound. The M perturbed evaluations are independent, so they are
    farmed out to a joblib pool whose width is the parameter count (M). During
    the polish this beats parallelizing the per-frame loop: M perturbations
    overlap in one batch instead of running the ~M sequential gradient
    evaluations one after another.
    """
    from joblib import Parallel, delayed

    M     = x.size
    steps = np.empty(M)
    pts   = []
    for i in range(M):
        lo, hi = bounds[i]
        h = eps * max(1.0, abs(x[i]))
        if x[i] + h > hi:        # forward step would leave the box → step back
            h = -h
        steps[i] = h
        xp = x.copy()
        xp[i] += h
        pts.append(xp)

    vals = Parallel(n_jobs=n_jobs)(delayed(loss_fn)(p) for p in pts)
    return (np.asarray(vals, dtype=float) - f0) / steps


# ──────────────────────────────────────────────
# Reference scales + residuals for the least-squares optimizer
# ──────────────────────────────────────────────

def reference_scales(
    frames: list[Frame],
    wE:     np.ndarray | None = None,
    gid:    np.ndarray | None = None,
) -> tuple[float, float, float]:
    """
    Energy/force/stress scales used to normalize the loss so w_E/w_F/w_S are
    commensurable. ``norm_E`` is the *within-ref_group* weighted spread of the
    reference per-atom energies (each group's weighted mean removed first) — so a
    large between-group reference gap (AIMD vs DFT) does not inflate it and crush
    the meV-scale double well; ``norm_F`` = RMS of all reference force components;
    ``norm_S`` = RMS of the reference stress (eV/Å³, as normalized by the readers)
    over frames that carry stress. All fall back to 1.0 if degenerate.

    With one group + uniform weights, ``norm_E`` reduces to ``std(e_pa)`` exactly.
    """
    n     = len(frames)
    e_pa  = np.array([f.energy / len(f.species) for f in frames], dtype=float)
    if wE is None:  wE  = np.ones(n)
    if gid is None: gid = np.zeros(n, dtype=np.int64)

    de = e_pa.copy()
    for g in np.unique(gid):
        m = gid == g
        if wE[m].sum() > 0:
            de[m] -= np.average(de[m], weights=wE[m])
    norm_E = float(np.sqrt(np.sum(wE * de ** 2) / np.sum(wE))) if wE.sum() > 0 else float(np.std(e_pa))

    f_all  = np.concatenate([np.asarray(f.forces, dtype=float).ravel() for f in frames])
    norm_F = float(np.sqrt(np.mean(f_all ** 2)))

    s_list = [np.asarray(f.stress, dtype=float).ravel()
              for f in frames if f.stress is not None]
    norm_S = float(np.sqrt(np.mean(np.concatenate(s_list) ** 2))) if s_list else 1.0

    return (norm_E or 1.0), (norm_F or 1.0), (norm_S or 1.0)


def build_frame_weights(frames: list[Frame], weight_normalization: str = "count"):
    """Per-frame ref_group ids + per-channel weight arrays from Frame metadata.

    ``count`` normalization (default) sets each frame's per-channel weight to
    ``weight_X / N_{ref_group}`` — i.e. ``weight_X`` is a group-TOTAL share split
    across that group's frames, so the static:MD balance is invariant to MD
    stride/subsampling. ``none`` uses the raw per-frame ``weight_X``. With one
    group + uniform weights both reduce to flat weights (a no-op).

    Returns ``(gid, wE, wF, wS, groups)`` where ``groups`` is the ordered unique
    ref_group labels (gid indexes into it).
    """
    labels = [f.ref_group for f in frames]
    groups = list(dict.fromkeys(labels))
    gindex = {g: i for i, g in enumerate(groups)}
    gid = np.array([gindex[g] for g in labels], dtype=np.int64)
    wE  = np.array([f.weight_E for f in frames], dtype=float)
    wF  = np.array([f.weight_F for f in frames], dtype=float)
    wS  = np.array([f.weight_S for f in frames], dtype=float)
    if weight_normalization == "count":
        from collections import Counter
        ng    = Counter(labels)
        denom = np.array([ng[g] for g in labels], dtype=float)
        wE, wF, wS = wE / denom, wF / denom, wS / denom
    return gid, wE, wF, wS, groups


def _frame_ef_residual(bvff: BVFF, frame: Frame) -> tuple[float, float, np.ndarray]:
    """Per-frame pieces for the least-squares residual vector: predicted and
    reference per-atom energy, and the flattened force-component error
    (f_pred − f_ref). Module-level so joblib can pickle it."""
    n = len(frame.species)
    e, f = bvff.energy_and_forces(frame.lattice, frame.species, frame.positions)
    return float(e / n), float(frame.energy / n), (f - frame.forces).ravel()


def _residuals(
    x:            np.ndarray,
    keys:         list[str],
    params:       Parameters,
    neutral_dep:  str | None,
    counts:       dict[str, int],
    bvff_builder: Callable[[Parameters], BVFF],
    frames:       list[Frame],
    w_E:          float,
    w_F:          float,
    norm_E:       float,
    norm_F:       float,
    n_jobs:       int = 1,
    wE:           np.ndarray | None = None,
    wF:           np.ndarray | None = None,
    gid:          np.ndarray | None = None,
    lambda_reg:   float = 0.0,
    x0_seed:      np.ndarray | None = None,
) -> np.ndarray:
    """
    Residual vector for ``scipy.optimize.least_squares`` (minimizes ½‖r‖²).
    Energy block + per-frame force blocks, scaled so that

        ‖r‖² = w_E·(RMSE_E/norm_E)² + w_F·(RMSE_F/norm_F)²  (+ Tikhonov)

    i.e. the smooth least-squares analogue of ``compute_loss``, sharing its
    per-ref_group energy offset, normalization, and per-channel/per-frame
    weights. Note the objectives are *analogues*, not identical: least_squares
    minimizes the weighted sum of squared block-RMSEs while ``compute_loss``
    sums the block RMSEs linearly, so the E:F trade-off near the optimum can
    differ slightly (the reported best loss is always recomputed with
    ``compute_loss``). The per-frame force block is
    scaled by ``sqrt(wF_i/(Σ wF · c_i))`` with ``c_i = 3·N_i`` so it reproduces
    ``compute_loss``'s per-frame-mean force MSE even at mixed atom counts. With
    one group + uniform weights this reduces to the previous concatenated form.
    Non-finite residuals (parameter blow-ups) map to a large finite value.
    """
    p    = vector_to_params(x, keys, params, neutral_dep, counts)
    bvff = bvff_builder(p)

    if n_jobs != 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(
            delayed(_frame_ef_residual)(bvff, fr) for fr in frames
        )
    else:
        res = [_frame_ef_residual(bvff, fr) for fr in frames]

    n_fr   = len(res)
    e_bvff = np.array([r[0] for r in res])
    e_ref  = np.array([r[1] for r in res])
    rho    = [np.asarray(r[2], dtype=float) for r in res]      # per-frame force-error vectors

    if wE is None:  wE  = np.ones(n_fr)
    if wF is None:  wF  = np.ones(n_fr)
    if gid is None: gid = np.zeros(n_fr, dtype=np.int64)

    de = e_bvff - e_ref
    for g in np.unique(gid):                                   # per-ref_group offset
        m = gid == g
        if wE[m].sum() > 0:
            de[m] -= np.average(de[m], weights=wE[m])

    sumE = wE.sum() if wE.sum() > 0 else 1.0
    sumF = wF.sum() if wF.sum() > 0 else 1.0
    e_block = (np.sqrt(w_E) / norm_E) * np.sqrt(wE / sumE) * de
    f_blocks = []
    for i, ri in enumerate(rho):
        ci = max(1, ri.size)
        f_blocks.append((np.sqrt(w_F) / norm_F) * np.sqrt(wF[i] / (sumF * ci)) * ri)
    f_block = np.concatenate(f_blocks) if f_blocks else np.zeros(0)

    blocks = [e_block, f_block]
    if lambda_reg > 0 and x0_seed is not None:                 # Tikhonov pull toward the seed
        scale = np.maximum(np.abs(x0_seed), 1e-8)
        # /x.size so this block's sum-of-squares equals the MEAN-semantics
        # penalty used by the anneal objective and the polish (lambda_reg *
        # mean(((x-x0)/s)^2)) — otherwise the same lambda_reg is a factor of
        # n_params stronger in the least_squares path than in the others.
        blocks.append(np.sqrt(lambda_reg / max(1, x.size)) * (x - x0_seed) / scale)

    out = np.concatenate(blocks)
    # least_squares cannot handle inf/nan; map them to a large finite residual.
    return np.where(np.isfinite(out), out, np.sqrt(_LOSS_PENALTY))


def _residuals_z(z, transform, *args):
    """Optimizer-space wrapper: the residual (incl. Tikhonov anchor) is
    defined on physical values. Module-level so joblib can ship it to
    multi-start workers."""
    return _residuals(transform.from_opt(z), *args)


def _frame_grad_cols(bvff: BVFF, frame: Frame, grad_keys: list[str]):
    """Per-frame analytic Jacobian pieces: per-atom energy gradient row (M_a,)
    and flattened force-gradient block (3N, M_a) over the analytic keys.
    Module-level so joblib can pickle it."""
    g = bvff.param_grads(frame.lattice, frame.species, frame.positions)
    n = len(frame.species)
    dE = np.array([g[k][0] / n for k in grad_keys])
    dF = np.stack([g[k][1].ravel() for k in grad_keys], axis=1) \
        if grad_keys else np.zeros((3 * n, 0))
    return dE, dF


def _jacobian_z(
    z:            np.ndarray,
    transform:    "ParamTransform",
    keys:         list[str],
    params:       Parameters,
    neutral_dep:  str | None,
    counts:       dict[str, int],
    bvff_builder: Callable[[Parameters], BVFF],
    frames:       list[Frame],
    w_E:          float,
    w_F:          float,
    norm_E:       float,
    norm_F:       float,
    n_jobs:       int = 1,
    wE:           np.ndarray | None = None,
    wF:           np.ndarray | None = None,
    gid:          np.ndarray | None = None,
    lambda_reg:   float = 0.0,
    x0_seed:      np.ndarray | None = None,
) -> np.ndarray:
    """
    Analytic Jacobian of ``_residuals_z`` (same argument list, same row
    layout: energy block, per-frame force blocks, optional Tikhonov block).

    Columns come from the terms' ``param_grads`` (∂E/∂θ, ∂F/∂θ per frame);
    the energy rows get the same per-ref_group weighted centering as the
    residual (the optimal offset moves with θ), the force rows the same
    per-frame scaling, and the whole matrix is chained into optimizer space
    (dx/dz = ln10·x for log-space magnitudes, 1 otherwise). Vector keys no
    term claims analytically — the Coulomb/Ewald charges, only present with
    fit_charges=1 — fall back to forward-difference columns in z-space.

    Replaces scipy's jac="2-point", which costs (M+1) full E+F passes per
    Jacobian; this costs ~one param_grads pass (a few energy_and_forces
    equivalents) plus one FD pass per charge key.
    """
    x    = transform.from_opt(z)
    p    = vector_to_params(x, keys, params, neutral_dep, counts)
    bvff = bvff_builder(p)

    # Which keys the terms cover analytically (stable across frames: every
    # term pre-fills all of its parameter keys, zeros included).
    probe = bvff.param_grads(frames[0].lattice, frames[0].species,
                             frames[0].positions)
    grad_keys = [k for k in keys if k in probe]
    fd_keys   = [k for k in keys if k not in probe]
    col_of    = {k: i for i, k in enumerate(keys)}

    if n_jobs != 1:
        from joblib import Parallel, delayed
        per_frame = Parallel(n_jobs=n_jobs)(
            delayed(_frame_grad_cols)(bvff, fr, grad_keys) for fr in frames
        )
    else:
        per_frame = [_frame_grad_cols(bvff, fr, grad_keys) for fr in frames]

    n_fr = len(frames)
    Ma   = len(grad_keys)
    if wE is None:  wE  = np.ones(n_fr)
    if wF is None:  wF  = np.ones(n_fr)
    if gid is None: gid = np.zeros(n_fr, dtype=np.int64)
    sumE = wE.sum() if wE.sum() > 0 else 1.0
    sumF = wF.sum() if wF.sum() > 0 else 1.0

    # Energy block: center each column within each ref_group (weighted), then
    # scale — mirroring _residuals' offset removal exactly.
    G_E = np.array([pf[0] for pf in per_frame])            # (n_fr, Ma)
    for g in np.unique(gid):
        m = gid == g
        if wE[m].sum() > 0 and Ma:
            G_E[m] -= np.average(G_E[m], axis=0, weights=wE[m])
    a_row = (np.sqrt(w_E) / norm_E) * np.sqrt(wE / sumE)   # (n_fr,)
    J_E   = a_row[:, None] * G_E

    # Force blocks: per-frame scaling b_i, rows stacked in frame order.
    F_rows = []
    for i, (_, dF) in enumerate(per_frame):
        ci = max(1, dF.shape[0])
        b  = (np.sqrt(w_F) / norm_F) * np.sqrt(wF[i] / (sumF * ci))
        F_rows.append(b * dF)
    J_F = np.vstack(F_rows) if F_rows else np.zeros((0, Ma))

    n_rows = n_fr + J_F.shape[0]
    tik = lambda_reg > 0 and x0_seed is not None
    if tik:
        n_rows += len(keys)
    J = np.zeros((n_rows, len(keys)))
    if Ma:
        cols = [col_of[k] for k in grad_keys]
        J[:n_fr, cols]                    = J_E
        J[n_fr:n_fr + J_F.shape[0], cols] = J_F
    if tik:
        scale = np.maximum(np.abs(x0_seed), 1e-8)
        J[n_fr + J_F.shape[0]:, :] = np.diag(
            np.sqrt(lambda_reg / max(1, len(keys))) / scale
        )

    # Chain rule physical → optimizer space: dx/dz = ln10·x on log columns.
    dxdz = np.where(transform.is_log, np.log(10.0) * x, 1.0)
    J   *= dxdz[None, :]

    # FD fallback columns (z-space) for keys without analytic coverage.
    if fd_keys:
        res_args = (keys, params, neutral_dep, counts, bvff_builder, frames,
                    w_E, w_F, norm_E, norm_F, n_jobs, wE, wF, gid,
                    lambda_reg, x0_seed)
        r0 = _residuals_z(z, transform, *res_args)
        zb = transform.opt_bounds()
        for k in fd_keys:
            c  = col_of[k]
            h  = 1e-6 * max(1.0, abs(z[c]))
            if z[c] + h > zb[c][1]:
                h = -h
            zp = z.copy()
            zp[c] += h
            J[:, c] = (_residuals_z(zp, transform, *res_args) - r0) / h

    return np.where(np.isfinite(J), J, 0.0)


def _single_least_squares(
    z0:        np.ndarray,
    zbounds:   list[tuple[float, float]],
    maxiter:   int,
    jac_mode:  str,
    transform: "ParamTransform",
    res_args:  tuple,
):
    """One bounded trust-region run from one start (module-level so a
    multi-start ensemble can fan out over a joblib pool). Returns
    (z_best, cost, nfev, status)."""
    from scipy.optimize import least_squares
    jac = _jacobian_z if jac_mode == "analytic" else "2-point"
    result = least_squares(
        _residuals_z, z0, bounds=(np.array([b[0] for b in zbounds]),
                                  np.array([b[1] for b in zbounds])),
        method="trf", jac=jac, x_scale="jac", max_nfev=maxiter,
        args=(transform, *res_args),
    )
    return result.x, float(result.cost), int(result.nfev), int(result.status)


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
    charge_neutral:    bool  = True,
    n_jobs:            int   = 1,
    n_starts:          int   = 1,
    optimizer:         str   = "least_squares",
    jac:               str   = "analytic",
    fit_charges:       bool  = False,
    weight_normalization: str = "count",
    lambda_reg:        float = 0.0,
    log_space:         bool  = True,
    diagnostics:       dict | None = None,
) -> tuple[Parameters, float]:
    """
    Fit BVFF parameters with Simulated Annealing (scipy ``dual_annealing``),
    optionally followed by an L-BFGS-B local polish.

    The global stage explores the bounded parameter space and is robust to the
    rugged, multi-modal BVFF loss surface. The optional polish stage then runs
    a gradient-based (numerical-gradient) L-BFGS-B refinement from the global
    best; the better of the two results is kept, so polish can only help.

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
        charge_neutral:     when use_coulomb is on, fit all charges but one and
                            fix the remaining (most abundant) species' charge by
                            the neutrality condition, so every candidate cell is
                            neutral — required for physical Ewald/Coulomb energies.
        n_jobs:             process budget for joblib parallelism (1 = serial,
                            -1 = all cores). Spent on whichever single level is
                            active so cores are never oversubscribed: the
                            per-frame loss loop for a single start, the ensemble
                            of starts when n_starts > 1, and the polish's
                            finite-difference gradient.
        n_starts:           number of independent optimizer runs for a
                            multi-start ensemble; the best result is kept.
                            1 = single start (frame loop uses n_jobs). With
                            n_starts > 1 the starts run in parallel (n_jobs) and
                            each start's frame loop is serial, so the budget goes
                            to the ensemble instead — better core utilization and
                            a more robust escape from shallow basins. For
                            optimizer="anneal" the starts differ by RNG seed; for
                            "least_squares" start 0 is the physical seed and the
                            rest are bounded perturbations of it in optimizer
                            space (±25% of each bound span).
        jac:                least_squares Jacobian: "analytic" (default —
                            assembled from the terms' param_grads, exact and
                            ~n_params× cheaper per iteration) or "fd"
                            (scipy 2-point finite differences).
        log_space:          optimize magnitude parameters (repulsive B,
                            Buckingham A/C, BV S, BVV D) as log10(value) — see
                            ParamTransform. Collapse-to-zero becomes a visible
                            walk to the log lower bound, and the FD Jacobian is
                            conditioned across the orders of magnitude these
                            prefactors span. Reported values/bounds diagnostics
                            stay in physical space.
        diagnostics:        optional dict the fit fills in place with
                            post-fit diagnostics (bound_flags from
                            bound_saturation_report, n_parameters, optimizer,
                            best_loss, ...) so the caller can persist them
                            without a return-signature change.

    Returns:
        fitted_params: optimized Parameters dataclass
        best_loss:     final loss value
    """
    # Charge neutrality: pick the most abundant charged species as the
    # "dependent" one. Its charge is reconstructed from the others (see
    # vector_to_params) instead of being fit, keeping Σ q_i n_i = 0. Most
    # abundant → largest denominator → numerically stable.
    counts = dict(Counter(train_frames[0].species))
    neutral_dep = None
    if not fit_charges and controls_potentials.use_coulomb:
        logger.info(
            "Charges held fixed at their parameters.toml values (fit_charges=0): "
            "recommended for a bond-valence FF — fitting charges against E+F is "
            "under-determined and collapses them toward zero. Seed charges should "
            "be physical (e.g. formal/Born values) and cell-neutral."
        )
    if fit_charges and charge_neutral and controls_potentials.use_coulomb and params.coulomb.charges:
        neutral_dep = max(
            params.coulomb.charges, key=lambda s: counts.get(s, 0)
        )
        logger.info(
            f"Charge neutrality enforced: '{neutral_dep}' charge is fixed by "
            f"Σ q_i n_i = 0 (n_{neutral_dep}={counts.get(neutral_dep)}); the other "
            f"charges are fit freely."
        )

    x0, keys = params_to_vector(params, controls_potentials, neutral_dep, fit_charges)
    bounds   = build_bounds(keys)

    # Optimizer-space transform: magnitude parameters are fit as log10(value)
    # (identity when log_space=False). All z-suffixed variables below live in
    # the optimizer space; physical values are recovered via transform.from_opt.
    transform = ParamTransform(keys, bounds, log_space=log_space)
    zbounds   = transform.opt_bounds()
    if log_space and transform.is_log.any():
        logger.info(
            f"Log-space optimization for {int(transform.is_log.sum())}/{len(keys)} "
            f"magnitude parameters (repulsive B / Buckingham A,C / BV S / BVV D)."
        )

    # The seed is clipped into bounds and projected through the optimizer
    # transform (a deliberate 0.0 on a log-space magnitude becomes its
    # effective floor). x0_seed is BOTH the optimization start and the
    # Tikhonov anchor, so the anchor is always reachable and the penalty is
    # exactly 0 at the start — anchoring at an unreachable raw 0.0 below the
    # log floor would add an irreducible ~(floor/1e-8)² penalty that breaks
    # early stopping and cross-run loss comparison.
    lb_all  = np.array([b[0] for b in bounds])
    ub_all  = np.array([b[1] for b in bounds])
    x0_seed = transform.from_opt(transform.to_opt(np.clip(x0, lb_all, ub_all)))

    # Project the starting params onto the neutral subspace so the reported
    # initial loss is consistent with the constraint the fit will enforce.
    params  = vector_to_params(x0_seed, keys, params, neutral_dep, counts)

    # Multi-source weights: per-frame ref_group ids + per-channel weight arrays
    # (constant across iterations). With one group + unit weights these are flat,
    # reproducing single-offset, equal-weight behavior.
    gid, wE, wF, wS, groups = build_frame_weights(train_frames, weight_normalization)

    # Reference scales (group-aware norm_E within each ref_group + norm_S) so the
    # energy/force/stress blocks are commensurable — see reference_scales.
    norm_E, norm_F, norm_S = reference_scales(train_frames, wE, gid)

    n_starts = max(1, n_starts)
    logger.info(f"Fitting {len(keys)} parameters via '{optimizer}' ...")
    logger.info(f"Loss weights: w_E={w_E}, w_F={w_F}, w_S={w_S}")
    logger.info(f"Reference scales: norm_E={norm_E:.6f} eV/atom, norm_F={norm_F:.6f} eV/Å, norm_S={norm_S:.6g} eV/Å³")
    if len(groups) > 1 or any(g != "default" for g in groups):
        from collections import Counter as _C
        gc = _C(f.ref_group for f in train_frames)
        logger.info(f"Reference groups (per-group energy offset): {dict(gc)}")
        for g, ng in gc.items():
            if ng < 2:
                logger.warning(
                    f"  ref_group '{g}' has only {ng} train frame(s): its energy "
                    f"residual is ~0 after offset removal (forces/stress still fit). "
                    f"Merge it into its same-DFT-setup group if unintended."
                )
        # Which groups carry stress (for use_stress visibility).
        sg = sorted({f.ref_group for f in train_frames if f.stress is not None})
        if sg:
            logger.info(f"  stress present in ref_groups: {sg}")
    if lambda_reg > 0:
        logger.info(f"Tikhonov regularization toward seed: lambda_reg={lambda_reg}")
    if n_starts > 1:
        logger.info(
            f"Multi-start ensemble: {n_starts} independent SA runs "
            f"(seeds {seed}..{seed + n_starts - 1}) in parallel over n_jobs={n_jobs}; "
            f"each run's per-frame loop is serial to avoid oversubscription."
        )
    elif n_jobs != 1:
        logger.info(
            f"Per-frame loss loop parallelized over n_jobs={n_jobs} "
            f"(joblib) for {len(train_frames)} training frames."
        )

    # Show the starting-point loss before optimization begins so the user
    # can see "fitting is alive" immediately. The progress bar streams per-frame
    # progress; every SA evaluation also gets its own bar (see objective).
    logger.info(f"  Computing initial loss on {len(train_frames)} training frames (this can take a while) ...")
    t_fit_start = time.time()
    initial_loss = compute_loss(
        bvff_builder(params), train_frames, w_E, w_F, w_S, has_stress,
        use_stress=use_stress, progress=True, progress_label="initial-loss",
        norm_E=norm_E, norm_F=norm_F, norm_S=norm_S, wE=wE, wF=wF, wS=wS, gid=gid,
    )
    logger.info(f"  Initial loss = {initial_loss:.6f} (1 evaluation took {time.time() - t_fit_start:.2f}s)")
    eval_s = time.time() - t_fit_start
    if eval_s > 5.0:
        logger.info(
            f"  Note: one loss evaluation costs ~{eval_s:.1f}s; SA will run many of these. "
            f"Consider raising frame_start / lowering train_ratio / using stride to subsample frames."
        )

    if target_loss > 0:
        logger.info(f"  Early stop: target_loss = {target_loss:.6f}")
    if patience > 0:
        logger.info(f"  Early stop: patience    = {patience} evaluations without improvement")

    if optimizer == "least_squares":
        # ── Seeded local least-squares (trust-region) ────────────────────────
        # The default. The BVFF loss surface is rugged, but parameters.toml gives
        # a physically sensible seed, so a gradient-based trust-region solver that
        # *exploits* that seed converges far better than dual_annealing, which
        # samples the whole bounds box and ignores x0. Minimizes ½‖_residuals‖²,
        # i.e. the squared scale-normalized E/F loss (energies up to a constant).
        if any(f.stress is not None for f in train_frames):
            logger.warning(
                "optimizer='least_squares' fits energy+forces only; the stress "
                "term is excluded from the residual even though some frames carry "
                "stress (use optimizer='anneal' to include stress in the loss)."
            )
        if jac == "analytic":
            logger.info(
                "Jacobian: analytic parameter gradients (param_grads) — exact "
                "columns, ~one gradient pass per iteration instead of the "
                "(n_params+1) finite-difference residual passes."
            )
        else:
            logger.info("Jacobian: 2-point finite differences (fitting.jac='fd').")
        t_ls = time.time()
        z0  = transform.to_opt(x0_seed)
        zlb = np.array([b[0] for b in zbounds])
        zub = np.array([b[1] for b in zbounds])

        if n_starts == 1:
            starts = [z0]
        else:
            # Start 0 is the physical seed; the rest perturb it uniformly by
            # ±25% of each optimizer-space bound span (reproducible, in-bounds).
            # In log space that spread covers decades of the magnitude params —
            # a genuine basin ensemble, not gradient noise.
            starts = [z0]
            span = zub - zlb
            for i in range(1, n_starts):
                rng = np.random.default_rng(seed + i)
                starts.append(np.clip(
                    z0 + rng.uniform(-0.25, 0.25, z0.size) * span, zlb, zub
                ))
            logger.info(
                f"Multi-start least-squares: {n_starts} starts (seed + "
                f"{n_starts - 1} perturbed, seeds {seed + 1}..{seed + n_starts - 1}) "
                f"in parallel over n_jobs={n_jobs}; each start's frame loop is "
                f"serial to avoid oversubscription."
            )

        res_args_of = lambda frame_jobs: (
            keys, params, neutral_dep, counts, bvff_builder, train_frames,
            w_E, w_F, norm_E, norm_F, frame_jobs, wE, wF, gid, lambda_reg, x0_seed,
        )
        if len(starts) == 1:
            runs = [_single_least_squares(
                starts[0], zbounds, maxiter, jac, transform, res_args_of(n_jobs),
            )]
        else:
            from joblib import Parallel, delayed
            runs = Parallel(n_jobs=n_jobs)(
                delayed(_single_least_squares)(
                    zs, zbounds, maxiter, jac, transform, res_args_of(1),
                ) for zs in starts
            )
            for i, (_, cost, nfev, status) in enumerate(runs):
                logger.info(
                    f"  start {i}{' (seed point)' if i == 0 else '':s}: "
                    f"cost={cost:.6g} | {nfev} evals | status={status}"
                )

        best_idx  = int(np.argmin([r[1] for r in runs]))
        best_z    = runs[best_idx][0]
        best_loss = compute_loss(
            bvff_builder(vector_to_params(transform.from_opt(best_z), keys, params,
                                          neutral_dep, counts)),
            train_frames, w_E, w_F, w_S, has_stress, use_stress=use_stress,
            norm_E=norm_E, norm_F=norm_F, norm_S=norm_S, wE=wE, wF=wF, wS=wS, gid=gid,
            n_jobs=n_jobs,
        )
        logger.info(
            f"Least-squares done in {time.time() - t_ls:.1f}s | "
            f"{sum(r[2] for r in runs)} residual evals over {len(runs)} start(s) "
            f"| best = start {best_idx} | loss={best_loss:.6f} "
            f"(initial {initial_loss:.6f})"
        )
    else:
        # ── Global stage: single start, or a parallel multi-start ensemble ────
        # The two modes are mutually exclusive in how they spend the n_jobs budget
        # so cores are never oversubscribed (see _single_anneal / the docstring).
        # x0_seed is already clipped into bounds, so dual_annealing never sees
        # an out-of-bounds start (it raises on one, unlike least_squares).
        anneal_args = dict(
            x0=transform.to_opt(x0_seed), bounds=zbounds, keys=keys, params=params,
            neutral_dep=neutral_dep,
            counts=counts, bvff_builder=bvff_builder, train_frames=train_frames,
            w_E=w_E, w_F=w_F, w_S=w_S, has_stress=has_stress, use_stress=use_stress,
            maxiter=maxiter, target_loss=target_loss, patience=patience,
            initial_loss=initial_loss, norm_E=norm_E, norm_F=norm_F, norm_S=norm_S,
            wE=wE, wF=wF, wS=wS, gid=gid, lambda_reg=lambda_reg, x0_seed=x0_seed,
            transform=transform,
        )

        if n_starts == 1:
            # Single start in the main process: full per-iteration logging streams,
            # and the per-frame loss loop uses the n_jobs budget.
            best_z, best_loss, n_evals, stop_reason = _single_anneal(
                seed=seed, n_jobs=n_jobs, verbose=True, **anneal_args
            )
            if stop_reason:
                logger.info(f"Early-stopped: {stop_reason}")
            logger.info(
                f"Global stage done in {time.time() - t_fit_start:.1f}s "
                f"| {n_evals} evaluations | best loss = {best_loss:.6f}"
            )
        else:
            # Ensemble: n_starts runs with distinct seeds, dispatched across the
            # n_jobs pool. Each run is serial-framed (n_jobs=1) and quiet
            # (verbose=False) since worker logging does not reach the main log.
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=n_jobs)(
                delayed(_single_anneal)(seed=seed + i, n_jobs=1, verbose=False, **anneal_args)
                for i in range(n_starts)
            )
            for i, (_, bl, ne, sr) in enumerate(results):
                logger.info(
                    f"  start {i} (seed {seed + i}): best loss = {bl:.6f} "
                    f"| {ne} evals | {sr or 'maxiter reached'}"
                )
            best_idx = int(np.argmin([r[1] for r in results]))
            best_z, best_loss = results[best_idx][0], float(results[best_idx][1])
            logger.info(
                f"Global stage done in {time.time() - t_fit_start:.1f}s "
                f"| {n_starts} starts | best = start {best_idx} (loss {best_loss:.6f})"
            )

        # ── Optional local polish: L-BFGS-B from the global best ──────────────
        # The value evaluations run serial-framed; the numerical gradient's M
        # perturbations are evaluated in parallel (_fd_grad_parallel) — a better
        # fit for a many-core node than the per-frame loop during polish. Kept
        # only if it improves on the global best.
        if polish:
            logger.info("Polishing best point with L-BFGS-B ...")
            t_polish = time.time()

            def loss_serial(z: np.ndarray) -> float:
                x_phys = transform.from_opt(z)
                p = vector_to_params(x_phys, keys, params, neutral_dep, counts)
                loss = compute_loss(
                    bvff_builder(p), train_frames, w_E, w_F, w_S, has_stress,
                    use_stress=use_stress, progress=False, n_jobs=1,
                    norm_E=norm_E, norm_F=norm_F, norm_S=norm_S,
                    wE=wE, wF=wF, wS=wS, gid=gid,
                )
                if lambda_reg > 0:
                    scale = np.maximum(np.abs(x0_seed), 1e-8)
                    loss += lambda_reg * float(np.mean(((x_phys - x0_seed) / scale) ** 2))
                return loss

            jac = None
            if n_jobs != 1:
                def jac(z: np.ndarray) -> np.ndarray:
                    return _fd_grad_parallel(z, loss_serial(z), loss_serial, zbounds, n_jobs)

            polish_res = minimize(
                loss_serial, best_z, method="L-BFGS-B", bounds=zbounds, jac=jac,
            )
            polish_loss = float(polish_res.fun)
            if polish_loss < best_loss:
                logger.info(
                    f"  Polish improved loss {best_loss:.6f} → {polish_loss:.6f} "
                    f"({time.time() - t_polish:.1f}s)"
                )
                best_z, best_loss = polish_res.x, polish_loss
            else:
                logger.info(
                    f"  Polish did not improve (kept {best_loss:.6f}, "
                    f"{time.time() - t_polish:.1f}s)"
                )

    best_x        = transform.from_opt(best_z)
    fitted_params = vector_to_params(best_x, keys, params, neutral_dep, counts)
    logger.info(
        f"Fitting completed in {time.time() - t_fit_start:.1f}s "
        f"| best loss = {best_loss:.6f}"
    )

    # Bound-saturation diagnostics (physical space): a fit whose parameters sit
    # on the bounds box is a degeneracy symptom that must be loud, not something
    # discovered three weeks later by reading fitted_parameters.toml digit by
    # digit (or by an exploded MD run missing its O-Ti repulsive wall).
    flags = bound_saturation_report(best_x, keys, bounds, transform=transform)
    if flags:
        logger.warning(
            f"{len(flags)}/{len(keys)} fitted parameters sit at/near a bound or "
            f"collapsed to ~0 — this is an identifiability/degeneracy symptom, "
            f"not a converged physical optimum:"
        )
        for fl in flags:
            logger.warning(
                f"  {fl['key']:28s} = {fl['value']:.6g}   "
                f"bounds [{fl['lower']:g}, {fl['upper']:g}]   ({fl['flag']})"
            )
        logger.warning(
            "  Consider: lambda_reg > 0 (Tikhonov pull toward the seed), a softer "
            "repulsion (use_buckingham=1), or explicitly freezing/removing the "
            "collapsed term instead of letting the optimizer zero it."
        )
    if diagnostics is not None:
        diagnostics.update({
            "optimizer":     optimizer,
            "jacobian":      jac if optimizer == "least_squares" else "n/a",
            "n_parameters":  len(keys),
            "n_starts":      int(n_starts),
            "log_space":     bool(log_space),
            "lambda_reg":    float(lambda_reg),
            "best_loss":     float(best_loss),
            "initial_loss":  float(initial_loss),
            "bound_flags":   flags,
        })

    return fitted_params, best_loss
