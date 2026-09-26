from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np
from scipy.optimize import dual_annealing, minimize

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
) -> float:
    """
    Compute weighted RMSE loss:
        Loss = w_E * RMSE_E + w_F * RMSE_F (+ w_S * RMSE_S if stress enabled)

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
    rmse_E = np.sqrt(np.mean(e_res ** 2))
    rmse_F = np.sqrt(np.mean(f_errors))

    loss = w_E * rmse_E + w_F * rmse_F

    if has_stress and s_errors:
        rmse_S = np.sqrt(np.mean(s_errors))
        loss  += w_S * rmse_S

    return float(loss)


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

    x0, keys = params_to_vector(params, controls_potentials, composition)
    bounds   = build_bounds(keys)

    logger.info(f"Fitting {len(keys)} parameters with Simulated Annealing ...")
    logger.info(f"Loss weights: w_E={w_E}, w_F={w_F}, w_S={w_S}")

    # Show the starting-point loss before optimization begins so the user
    # can see "fitting is alive" immediately. The progress bar streams per-frame
    # progress; every SA evaluation also gets its own bar (see objective).
    logger.info(f"  Computing initial loss on {len(train_frames)} training frames (this can take a while) ...")
    t_fit_start = time.time()
    initial_loss = compute_loss(
        bvff_builder(params), train_frames, w_E, w_F, w_S, has_stress,
        use_stress=use_stress, progress=True, progress_label="initial-loss",
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
            use_stress=use_stress, progress=False,
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
