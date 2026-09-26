from __future__ import annotations

import logging
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
from src.extensions.ewald import Ewald
from src.fitting import fit
from src.outputs import setup_logger, section, save_results
from scripts.analysis import run_analysis


# ──────────────────────────────────────────────
# Potential builder
# ──────────────────────────────────────────────

def build_bvff(controls: Controls, params: Parameters, logger: logging.Logger = None) -> BVFF:
    terms: list[Potential] = []
    cutoff = params.cutoff
    pc     = controls.potentials
    ext    = controls.extensions

    if pc.use_coulomb:
        if ext.use_ewald:
            ew = Ewald(
                charges  = params.coulomb.charges,
                cutoff   = ext.ewald_cutoff,
                accuracy = ext.ewald_accuracy,
            )
            terms.append(ew)
            if logger:
                logger.info(
                    f"Potential: Coulomb (Ewald, tin-foil) r_c={ew.cutoff:.2f} Å, "
                    f"δ={ew.accuracy:.0e} → α={ew.alpha:.4f} Å⁻¹, k_c={ew.kcut:.4f} Å⁻¹"
                )
        else:
            terms.append(Coulomb(
                charges = params.coulomb.charges,
                cutoff  = cutoff,
            ))
            if logger: logger.info("Potential: Coulomb (direct)")

    if pc.use_repulsive:
        terms.append(Repulsive(B=params.repulsive.B, cutoff=cutoff))
        if logger: logger.info("Potential: Repulsive")

    if pc.use_buckingham:
        buck_params = {
            pair: {"A": bp.A, "rho": bp.rho, "C": bp.C}
            for pair, bp in params.buckingham.pairs.items()
        }
        terms.append(Buckingham(params=buck_params, cutoff=cutoff))
        if logger: logger.info("Potential: Buckingham (Born-Mayer + dispersion)")

    if pc.use_BV:
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
        ))
        if logger: logger.info(f"Potential: Bond Valence (BV, form={pc.bv_form})")

    if pc.use_BVV:
        bvv_species = {
            atom: {"W0": sp.W0, "D": sp.D}
            for atom, sp in params.BVV.species.items()
        }
        pair_params = {
            pair: {"r0": p.r0, "C": p.C, "b": p.b}
            for pair, p in params.BV.pairs.items()
        }
        terms.append(BVV(
            species_params = bvv_species,
            pair_params    = pair_params,
            cutoff         = cutoff,
            form           = pc.bv_form,
        ))
        if logger: logger.info(f"Potential: Bond Valence Vector (BVV, form={pc.bv_form})")

    if pc.use_angle:
        terms.append(Angle(k=params.angle.k, cutoff=cutoff))
        if logger: logger.info("Potential: Angle")

    if not terms:
        raise ValueError("No potential terms enabled in controls.toml.")

    return BVFF(terms=terms, energy_ref=params.energy_ref)


# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────

def split_frames(data: Dataset, train_ratio: float, seed: int = 0):
    """
    Stratified train/test split keyed on each frame's source file.

    With one trajectory (temperature) per file, splitting the flat frame list
    by ratio would dump the whole hottest run into the test tail. Instead we
    group by `Frame.source` and apply `train_ratio` *within* each group, so
    every temperature is represented in both train and test in proportion.

    Within each temperature the frames are shuffled before splitting, so the
    test set is iid across the run rather than its temporally-late tail. The
    shuffle is reproducible: each group draws from its own stream seeded by
    `(seed, group_index)`, so the split is identical run-to-run for a given
    seed and dataset, and independent between temperatures.

    Frames without a source ("") collapse into one group, so single-file or
    legacy datasets still get a clean seeded shuffle.
    """
    groups: dict[str, list[Frame]] = {}
    for fr in data.frames:
        groups.setdefault(fr.source, []).append(fr)

    train_frames: list[Frame] = []
    test_frames:  list[Frame] = []
    for i, frames in enumerate(groups.values()):
        shuffled = list(frames)
        # str seed → random hashes it with SHA-512, so the stream is stable
        # across runs (unlike tuple/PYTHONHASHSEED) and independent per group.
        random.Random(f"{seed}:{i}").shuffle(shuffled)
        n_train = int(len(shuffled) * train_ratio)
        train_frames.extend(shuffled[:n_train])
        test_frames.extend(shuffled[n_train:])

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
    section(logger, "Step 1/6: Loading dataset")
    t0 = time.time()
    data = load_dataset(entries=controls.dataset, logger=logger)
    logger.info(f"Loaded {len(data.frames)} frames | {data.n_atoms} atoms | species: {data.species} | {time.time() - t0:.1f}s")

    # 4. Parse parameters.toml (auto-generate if not found)
    section(logger, "Step 2/6: Loading parameters")
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

    # 5. Train / test split (stratified per source file → balanced across temperatures)
    train_frames, test_frames = split_frames(data, controls.train_ratio, seed=controls.split_seed)
    logger.info(
        f"Train frames: {len(train_frames)} | Test frames: {len(test_frames)} "
        f"(stratified per source, shuffle seed={controls.split_seed})"
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
    section(logger, "Step 3/6: Parameter fitting (Simulated Annealing)")
    t0 = time.time()
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
    )
    logger.info(f"Best loss: {best_loss:.6f} | fitting took {time.time() - t0:.1f}s")

    # 8. Save fitted parameters
    section(logger, "Step 4/6: Saving fitted parameters")
    fitted_params_path = str(Path(controls.output_dir) / "fitted_parameters.toml")
    save_parameters(fitted_params, fitted_params_path)
    logger.info(f"Fitted parameters saved to {fitted_params_path}.")

    # 9. Build BVFF with fitted parameters
    bvff = build_bvff(controls, fitted_params, logger)

    # 10. Save results
    section(logger, "Step 5/6: Computing & saving predictions")
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

    # 11. Run analysis
    section(logger, "Step 6/6: Running analysis & plots")
    t0 = time.time()
    run_analysis(
        bvff         = bvff,
        train_frames = train_frames,
        test_frames  = test_frames,
        output_dir   = controls.output_dir,
    )
    logger.info(f"Analysis finished in {time.time() - t0:.1f}s")

    section(logger, f"BVFF completed in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
