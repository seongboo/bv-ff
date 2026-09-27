from __future__ import annotations

import logging
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

# Allow running as a script (e.g. `python3 ../../src/main.py`) by putting the
# project root on sys.path. When imported as `src.main` this is a no-op.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from parsers.controls_parser import parse_controls, Controls
from parsers.parameters_parser import parse_parameters, Parameters, save_parameters
from parsers.dataset import load_dataset, Dataset, Frame
from src.potentials import Potential, Coulomb, Repulsive, Buckingham, BV, BVV, Angle, BVFF
from src.extensions.ewald import Ewald, ewald_alpha
from src.fitting import fit
from src.outputs import setup_logger, section, save_results, save_provenance


# ──────────────────────────────────────────────
# Potential builder
# ──────────────────────────────────────────────

def build_bvff(
    controls: Controls,
    params:   Parameters,
    logger:   logging.Logger = None,
) -> BVFF:
    """Build the BVFF from controls + params. Ewald parameters are adaptive:
    alpha follows from the cutoff, and kmax is chosen per frame lattice, so
    datasets mixing cell sizes (e.g. a primitive DFT anchor + an AIMD
    supercell) get a converged reciprocal sum for every frame."""
    terms: list[Potential] = []
    cutoff = params.cutoff
    sw     = params.smooth_width
    pc     = controls.potentials
    ext    = controls.extensions
    if logger:
        logger.info(
            f"Cutoff smoothing: width={sw} Å (C² taper on all short-range terms)"
            if sw > 0 else
            "Cutoff smoothing: OFF (smooth_width=0) — MD energy is NOT "
            "conserved at cutoff crossings; fitting-only use recommended."
        )

    if pc.use_coulomb:
        if not params.coulomb.charges:
            raise ValueError(
                "use_coulomb is enabled but no [coulomb] charges are defined in "
                "parameters.toml. Add per-element charges or disable use_coulomb."
            )
        if ext.use_ewald:
            # alpha depends only on the cutoff; kmax=None lets the Ewald term
            # pick a converged kmax per lattice (a 2x2x2 supercell needs ~2x
            # the kmax of the primitive cell for the same accuracy).
            alpha = ewald_alpha(cutoff)
            terms.append(Ewald(
                charges = params.coulomb.charges,
                alpha   = alpha,
                kmax    = None,
                cutoff  = cutoff,
                epsilon = ext.ewald_epsilon,
            ))
            if logger:
                eps_str = "inf (tinfoil)" if math.isinf(ext.ewald_epsilon) \
                          else f"{ext.ewald_epsilon:g} (vacuum-like)"
                logger.info(
                    f"Potential: Coulomb (Ewald summation, alpha={alpha:.3f} 1/Å, "
                    f"kmax=auto per lattice, cutoff={cutoff} Å, "
                    f"surface epsilon={eps_str})"
                )
        else:
            terms.append(Coulomb(
                charges      = params.coulomb.charges,
                cutoff       = cutoff,
                smooth_width = sw,
            ))
            if logger: logger.info("Potential: Coulomb (direct)")

    if pc.use_repulsive:
        if not params.repulsive.B:
            raise ValueError(
                "use_repulsive is enabled but no [repulsive] B pair parameters are "
                "defined in parameters.toml."
            )
        terms.append(Repulsive(B=params.repulsive.B, cutoff=cutoff, smooth_width=sw))
        if logger: logger.info("Potential: Repulsive")

    if pc.use_buckingham:
        if not params.buckingham.pairs:
            raise ValueError(
                "use_buckingham is enabled but no [buckingham] pair parameters are "
                "defined in parameters.toml."
            )
        buck_params = {
            pair: {"A": bp.A, "rho": bp.rho, "C": bp.C}
            for pair, bp in params.buckingham.pairs.items()
        }
        terms.append(Buckingham(params=buck_params, cutoff=cutoff, smooth_width=sw))
        if logger: logger.info("Potential: Buckingham (Born-Mayer + dispersion, inner guard on)")

    if pc.use_BV:
        if not params.BV.species or not params.BV.pairs:
            raise ValueError(
                "use_BV is enabled but [BV.species] and/or [BV.pairs] are empty in "
                "parameters.toml — the BV term would contribute nothing."
            )
        species_params = {
            atom: {"V0": sp.V0, "S": sp.S}
            for atom, sp in params.BV.species.items()
        }
        pair_params = {
            pair: {"r0": p.r0, "C": p.C, "b": p.b}
            for pair, p in params.BV.pairs.items()
        }
        terms.append(BV(
            species_params = species_params,
            pair_params    = pair_params,
            cutoff         = cutoff,
            form           = pc.bv_form,
            smooth_width   = sw,
        ))
        if logger: logger.info(f"Potential: Bond Valence (BV, form={pc.bv_form})")

    if pc.use_BVV:
        if not params.BVV.species or not params.BVV.pairs:
            raise ValueError(
                "use_BVV is enabled but [BVV] species and/or BVV pair parameters are "
                "empty in parameters.toml — the BVV term would be degenerate (zero "
                "forces). Define [BVV.pairs] or enable use_BV so they can be inherited."
            )
        bvv_species = {
            atom: {"W0": sp.W0, "D": sp.D}
            for atom, sp in params.BVV.species.items()
        }
        # BVV uses its own pair parameters. The parser inherits a copy of BV's
        # pairs when [BVV.pairs] is omitted, so BVV no longer depends on use_BV.
        pair_params = {
            pair: {"r0": p.r0, "C": p.C, "b": p.b}
            for pair, p in params.BVV.pairs.items()
        }
        terms.append(BVV(
            species_params = bvv_species,
            pair_params    = pair_params,
            cutoff         = cutoff,
            form           = pc.bv_form,
            smooth_width   = sw,
        ))
        if logger: logger.info(f"Potential: Bond Valence Vector (BVV, form={pc.bv_form})")

    if pc.use_angle:
        terms.append(Angle(k=params.angle.k, cutoff=cutoff, smooth_width=sw))
        if logger: logger.info("Potential: Angle")

    if any(abs(v) > 0.0 for v in ext.efield):
        if not params.coulomb.charges:
            raise ValueError(
                "extensions.efield is nonzero but no [coulomb] charges are "
                "defined — the field couples to the rigid-ion charges."
            )
        from src.extensions.efield import EField
        terms.append(EField(charges=params.coulomb.charges, field=ext.efield))
        if logger:
            logger.warning(
                f"External E-field active: E = {tuple(ext.efield)} V/Å (couples "
                f"f_i = q_i·E). Analysis-only — do NOT fit against field-free "
                f"DFT data with a field on."
            )

    if not terms:
        raise ValueError("No potential terms enabled in controls.toml.")

    return BVFF(terms=terms)


# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────

def split_frames(data: Dataset, train_ratio: float, seed: int = 0,
                 mode: str = "random"):
    """
    Stratified train/test split keyed on each frame's source file.

    With one trajectory (temperature) per file, splitting the flat frame list
    by ratio would dump the whole hottest run into the test tail. Instead we
    group by `Frame.source` and apply `train_ratio` *within* each group, so
    every temperature is represented in both train and test in proportion.

    ``mode`` controls how each group's cut is made:

    * ``"random"`` — frames are shuffled before splitting, so the test set is
      iid across the run. The shuffle is reproducible: each group draws from
      its own stream seeded by ``(seed, group_index)`` (str seed → random
      hashes it with SHA-512, stable across runs unlike tuple/PYTHONHASHSEED).
      CAVEAT: AIMD frames a few strides apart are strongly time-correlated, so
      every random test frame has near-duplicates in train — train≈test RMSE
      is then guaranteed by construction and is NOT evidence against
      overfitting. Use it as an optimizer sanity check.
    * ``"block"`` — the temporally LAST ``(1 - train_ratio)`` contiguous
      fraction of each trajectory is held out (load order preserved), so test
      frames are decorrelated from train. This is the honest generalization
      metric.

    Frames without a source ("") collapse into one group, so single-file or
    legacy datasets still get a clean split.

    Frames flagged ``split=False`` (e.g. a sparse DFT anchor entry) bypass the
    ratio cut and go entirely to train — otherwise a 1-frame anchor's
    ``int(1*0.8)=0`` would send it wholly to test and the fit would never see
    it. Frames flagged ``split="test"`` go entirely to test — holding out a
    whole file/condition (e.g. a temperature, or the BaTiO3 set) as an
    extrapolation test. After the split, every ref_group is asserted to retain
    at least one train frame.
    """
    if mode not in ("random", "block"):
        raise ValueError(f"split mode must be 'random' or 'block', got '{mode}'.")

    groups: dict[str, list[Frame]] = {}
    train_frames: list[Frame] = []
    test_frames:  list[Frame] = []
    for fr in data.frames:
        sp = getattr(fr, "split", True)
        if sp == "test":
            test_frames.append(fr)           # held-out condition: all to test
        elif sp is False or sp == "train":
            train_frames.append(fr)          # anchor: all to train
        else:
            groups.setdefault(fr.source, []).append(fr)

    for i, frames in enumerate(groups.values()):
        ordered = list(frames)
        if mode == "random":
            random.Random(f"{seed}:{i}").shuffle(ordered)
        # mode == "block": keep load (time) order — train on the head of the
        # trajectory, test on its tail.
        n_train = int(len(ordered) * train_ratio)
        if n_train == 0 and train_ratio > 0:
            logging.getLogger("bvff").warning(
                f"Source '{ordered[0].source or '<unknown>'}' has only "
                f"{len(ordered)} frame(s), so int({len(ordered)}*{train_ratio}) = 0 "
                f"and ALL its frames go to test — the fit never sees them. If it "
                f"is training data (e.g. a ground-state anchor), set split=false "
                f"on that dataset entry."
            )
        train_frames.extend(ordered[:n_train])
        test_frames.extend(ordered[n_train:])

    # Every ref_group must have ≥1 train frame (else its offset/weights are moot,
    # and a test-only ref_group's energy offset would be undefined — analysis
    # would silently inherit the global offset and distort the test RMSE).
    train_groups = {getattr(fr, "ref_group", "default") for fr in train_frames}
    all_groups   = {getattr(fr, "ref_group", "default") for fr in data.frames}
    missing = all_groups - train_groups
    if missing:
        raise ValueError(
            f"ref_group(s) {sorted(missing)} have no training frames after the "
            f"split. Set split=false on their dataset entries, add more frames, "
            f"or — for a split=\"test\" holdout — give the held-out entry the "
            f"same ref_group as a training entry with the same DFT setup."
        )

    return train_frames, test_frames


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    t_start = time.time()

    # 1. Parse controls.toml
    controls = parse_controls("controls.toml")

    # 2. Setup logger
    logger = setup_logger(controls.log_file, controls.output_dir)
    section(logger, "BVFF started")
    for e in controls.dataset:
        logger.info(
            f"dataset : {e.path}  (start={e.frame_start}, end={e.frame_end}, stride={e.stride})"
        )
    logger.info(f"task    : {controls.task}")

    # 3. Load dataset (vasprun.xml or extxyz; one or many files, per-file windows)
    section(logger, "Step 1/7: Loading dataset")
    t0 = time.time()
    data = load_dataset(entries=controls.dataset, logger=logger)
    logger.info(f"Loaded {len(data.frames)} frames | {data.n_atoms} atoms | species: {data.species} | {time.time() - t0:.1f}s")

    # 4. Parse parameters.toml (auto-generate if not found)
    section(logger, "Step 2/7: Loading parameters")
    pc = controls.potentials
    params = parse_parameters(
        filepath       = "parameters.toml",
        validate       = True,
        species        = data.species,
        use_coulomb    = pc.use_coulomb,
        use_repulsive  = pc.use_repulsive,
        use_BV         = pc.use_BV,
        use_BVV        = pc.use_BVV,
        use_angle      = pc.use_angle,
        use_buckingham = pc.use_buckingham,
        output_dir     = controls.output_dir,
    )
    logger.info("Parameters loaded.")

    # Charge neutrality: Ewald (and a meaningful direct Coulomb) assume a
    # net-neutral cell. Warn if the initial charges break neutrality — the fit
    # can also drift here, since charges are bounded independently.
    if pc.use_coulomb and params.coulomb.charges:
        counts = Counter(data.frames[0].species)
        net = sum(params.coulomb.charges.get(s, 0.0) * n for s, n in counts.items())
        if abs(net) > 1e-6:
            logger.warning(
                f"Cell is not charge-neutral: net charge = {net:+.4f} e "
                f"(Σ q_i n_i over {dict(counts)}). Ewald/Coulomb energies are "
                f"only physical for a neutral cell."
            )

    # 5. Train / test split (stratified per source file → balanced across temperatures)
    train_frames, test_frames = split_frames(
        data, controls.train_ratio, seed=controls.split_seed,
        mode=controls.split_mode,
    )
    logger.info(
        f"Train frames: {len(train_frames)} | Test frames: {len(test_frames)} "
        f"(stratified per source, split_mode={controls.split_mode}"
        + (f", shuffle seed={controls.split_seed})" if controls.split_mode == "random" else ")")
    )
    if controls.split_mode == "random":
        logger.info(
            "  Note: random within-trajectory splits of time-correlated AIMD frames "
            "make train≈test RMSE largely guaranteed by construction — use "
            "split_mode=\"block\" for an honest generalization metric."
        )
    tr = Counter(f.source for f in train_frames)
    te = Counter(f.source for f in test_frames)
    for src in dict.fromkeys(f.source for f in data.frames):
        logger.info(f"  {src or '<unknown>'}: train={tr.get(src, 0)} test={te.get(src, 0)}")

    # 6. Check stress availability
    has_stress = any(f.stress is not None for f in train_frames)
    if has_stress and controls.fitting.use_stress:
        logger.info("Stress data detected and use_stress=1: virial stress term included in loss.")
    elif has_stress:
        logger.info("Stress data detected but use_stress=0: stress term excluded from loss.")
    else:
        logger.info("Stress data not available: stress term excluded from loss.")

    # 7. Fitting
    section(logger, f"Step 3/7: Parameter fitting (optimizer={controls.fitting.optimizer})")
    t0 = time.time()
    fit_diagnostics: dict = {}
    fitted_params, best_loss = fit(
        bvff_builder        = lambda p: build_bvff(controls, p),
        params              = params,
        controls_potentials = controls.potentials,
        train_frames        = train_frames,
        w_E                 = controls.fitting.w_E,
        w_F                 = controls.fitting.w_F,
        w_S                 = controls.fitting.w_S,
        has_stress          = has_stress,
        use_stress          = controls.fitting.use_stress,
        polish              = controls.fitting.polish,
        maxiter             = controls.fitting.maxiter,
        target_loss         = controls.fitting.target_loss,
        patience            = controls.fitting.patience,
        charge_neutral      = controls.fitting.charge_neutral,
        n_jobs              = controls.fitting.n_jobs,
        n_starts            = controls.fitting.n_starts,
        optimizer           = controls.fitting.optimizer,
        jac                 = controls.fitting.jac,
        fit_charges         = controls.fitting.fit_charges,
        weight_normalization = controls.fitting.weight_normalization,
        lambda_reg          = controls.fitting.lambda_reg,
        log_space           = controls.fitting.log_space,
        diagnostics         = fit_diagnostics,
    )
    logger.info(f"Best loss: {best_loss:.6f} | fitting took {time.time() - t0:.1f}s")

    # 8. Save fitted parameters + fit diagnostics
    section(logger, "Step 4/7: Saving fitted parameters")
    fitted_params_path = str(Path(controls.output_dir) / "fitted_parameters.toml")
    save_parameters(fitted_params, fitted_params_path)
    logger.info(f"Fitted parameters saved to {fitted_params_path}.")

    # Bound-saturation flags etc. — persisted so a degenerate fit is
    # self-declaring in the run directory, not just a log line that scrolls by.
    if fit_diagnostics:
        import tomli_w
        diag_path = Path(controls.output_dir) / "fit_diagnostics.toml"
        with open(diag_path, "wb") as f:
            tomli_w.dump(fit_diagnostics, f)
        n_flag = len(fit_diagnostics.get("bound_flags", []))
        logger.info(
            f"Fit diagnostics saved to {diag_path} "
            f"({n_flag} bound-saturation flag(s))."
        )

    # Full run provenance (inputs, config, environment, source hashes, fit
    # outcome) — the answer to "what exactly produced this fit?".
    save_provenance(
        output_dir      = controls.output_dir,
        controls        = controls,
        train_frames    = train_frames,
        test_frames     = test_frames,
        fit_diagnostics = fit_diagnostics,
        params_file     = "parameters.toml",
        elapsed_s       = time.time() - t_start,
        logger          = logger,
    )

    # 9. Build BVFF with fitted parameters
    bvff = build_bvff(controls, fitted_params, logger)

    # 10. Save results
    section(logger, "Step 5/7: Computing & saving predictions")
    t0 = time.time()
    save_results(
        bvff         = bvff,
        output_dir   = controls.output_dir,
        task         = controls.task,
        train_frames = train_frames,
        test_frames  = test_frames,
        logger       = logger,
    )
    logger.info(f"Predictions saved in {time.time() - t0:.1f}s")

    # 11. Run analysis. Imported here, not at module scope: analysis pulls in
    # matplotlib, which a fitting-only run (e.g. a headless compute node)
    # should not need just to import this module.
    section(logger, "Step 6/7: Running analysis & plots")
    t0 = time.time()
    from scripts.analysis import run_analysis
    analysis_test = test_frames
    if not test_frames:
        # Legitimate config (every entry pinned split=false), but the analysis
        # reductions crash on empty arrays — reuse train and say so loudly.
        logger.warning(
            "Test set is empty (all dataset entries pinned to train): analysis "
            "reports TRAIN metrics in the test_* slots — they are NOT "
            "generalization metrics."
        )
        analysis_test = train_frames
    run_analysis(
        bvff         = bvff,
        train_frames = train_frames,
        test_frames  = analysis_test,
        output_dir   = controls.output_dir,
    )
    logger.info(f"Analysis finished in {time.time() - t0:.1f}s")

    # 12. Ferroelectric physics validation — the gating check RMSE cannot
    # provide. Persists ferroelectric_validation.toml + double_well.png next to
    # rmse_summary.txt so every fit run records its physics verdict. A FAIL is
    # logged loudly but does not abort: all artifacts are already saved, and a
    # failed potential is still worth inspecting.
    section(logger, "Step 7/7: Ferroelectric validation")
    if "Ti" not in data.species:
        logger.warning(
            "Ferroelectric validation skipped: the validation machinery assumes "
            "an ATiO3 perovskite (Ti B-site). Chemistry generalization is on the "
            "roadmap; validate non-titanates manually for now."
        )
    else:
        t0 = time.time()
        from scripts.ferroelectric import run_validation
        from src.calculator import BVFFCalculator
        # Retention needs a REPRESENTATIVE polar frame: prefer a ratio-split
        # trajectory frame — split=false anchors are appended first and may be
        # e.g. a 5-atom cubic-EOS cell where retention is meaningless.
        val_frame = next(
            (f for f in train_frames if getattr(f, "split", True) is True),
            train_frames[0] if train_frames else None,
        )
        fe_result = run_validation(
            BVFFCalculator(bvff),
            output_dir = controls.output_dir,
            charges    = fitted_params.coulomb.charges if pc.use_coulomb else None,
            data_frame = val_frame,
            logger     = logger,
        )
        if fe_result["passed"]:
            logger.info(f"Ferroelectric validation PASSED in {time.time() - t0:.1f}s.")
        else:
            logger.warning(
                f"Ferroelectric validation FAILED in {time.time() - t0:.1f}s: "
                + "; ".join(fe_result["reasons"])
                + " (see ferroelectric_validation.toml)"
            )

    section(logger, f"BVFF completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
