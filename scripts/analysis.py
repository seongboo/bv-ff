from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from parsers.dataset import Frame
from src.outputs import progress_iter
from src.potentials import BVFF

logger = logging.getLogger("bvff")


# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────

def collect_predictions(
    bvff:   BVFF,
    frames: list[Frame],
    label:  str = "frames",
) -> dict:
    """
    Collect BVFF predictions and AIMD reference values for all frames.

    Returns:
        dict with keys:
            e_bvff:  (N,) BVFF energies per atom (eV/atom)
            e_aimd:  (N,) AIMD energies per atom (eV/atom)
            f_bvff:  (N*n_atoms, 3) BVFF forces (eV/A)
            f_aimd:  (N*n_atoms, 3) AIMD forces (eV/A)
    """
    e_bvff_list, e_aimd_list = [], []
    f_bvff_list, f_aimd_list = [], []

    logger.info(f"  Collecting predictions on {len(frames)} {label} ...")
    t0 = time.time()

    for frame in progress_iter(frames, label=f"analysis/{label}"):
        n = len(frame.species)

        # Single pass: terms share their intermediates between the energy and
        # force kernels (half the cost of separate energy() + forces() calls).
        e_bvff, f_bvff = bvff.energy_and_forces(frame.lattice, frame.species, frame.positions)
        e_bvff_list.append(e_bvff / n)
        e_aimd_list.append(frame.energy / n)
        f_bvff_list.append(f_bvff)
        f_aimd_list.append(frame.forces)

    logger.info(f"  [{label}] predictions done in {time.time() - t0:.1f}s")

    return {
        "e_bvff": np.array(e_bvff_list),
        "e_aimd": np.array(e_aimd_list),
        "f_bvff": np.vstack(f_bvff_list),
        "f_aimd": np.vstack(f_aimd_list),
    }


# ──────────────────────────────────────────────
# RMSE summary
# ──────────────────────────────────────────────

def compute_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def print_rmse_summary(
    train_data: dict,
    test_data:  dict,
    logger:     logging.Logger,
) -> dict:
    """
    Compute and log RMSE for energy and forces on train/test sets.

    Returns:
        dict with RMSE values
    """
    rmse = {
        "train_E": compute_rmse(train_data["e_bvff"], train_data["e_aimd"]),
        "test_E":  compute_rmse(test_data["e_bvff"],  test_data["e_aimd"]),
        "train_F": compute_rmse(train_data["f_bvff"].ravel(), train_data["f_aimd"].ravel()),
        "test_F":  compute_rmse(test_data["f_bvff"].ravel(),  test_data["f_aimd"].ravel()),
    }

    logger.info("=" * 45)
    logger.info("  RMSE Summary")
    logger.info("=" * 45)
    logger.info(f"  Energy RMSE (train) : {rmse['train_E']:.6f} eV/atom (offset-removed)")
    logger.info(f"  Energy RMSE (test)  : {rmse['test_E']:.6f} eV/atom (offset-removed)")
    logger.info(f"  Force  RMSE (train) : {rmse['train_F']:.6f} eV/A")
    logger.info(f"  Force  RMSE (test)  : {rmse['test_F']:.6f} eV/A")
    logger.info("=" * 45)

    return rmse


# ──────────────────────────────────────────────
# Parity plot
# ──────────────────────────────────────────────

def plot_parity(
    train_data: dict,
    test_data:  dict,
    output_dir: str,
) -> None:
    """
    Generate parity plots for energy and forces (train and test).
    Saves to output_dir/parity_plot.png.
    """
    fig = plt.figure(figsize=(12, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    datasets = [
        ("train", train_data, gs[0, 0], gs[0, 1]),
        ("test",  test_data,  gs[1, 0], gs[1, 1]),
    ]

    for label, data, gs_e, gs_f in datasets:
        # Energy parity
        ax_e = fig.add_subplot(gs_e)
        ax_e.scatter(data["e_aimd"], data["e_bvff"], s=15, alpha=0.7, color="steelblue")
        lim_e = [
            min(data["e_aimd"].min(), data["e_bvff"].min()),
            max(data["e_aimd"].max(), data["e_bvff"].max()),
        ]
        ax_e.plot(lim_e, lim_e, "r--", lw=1.2, label="ideal")
        rmse_e = compute_rmse(data["e_bvff"], data["e_aimd"])
        ax_e.set_xlabel("AIMD Energy (eV/atom)")
        ax_e.set_ylabel("BVFF Energy (eV/atom)")
        ax_e.set_title(f"Energy Parity [{label}]\nRMSE = {rmse_e:.4f} eV/atom")
        ax_e.legend(fontsize=8)

        # Force parity
        ax_f = fig.add_subplot(gs_f)
        f_aimd_flat = data["f_aimd"].ravel()
        f_bvff_flat = data["f_bvff"].ravel()
        ax_f.scatter(f_aimd_flat, f_bvff_flat, s=5, alpha=0.3, color="darkorange")
        lim_f = [
            min(f_aimd_flat.min(), f_bvff_flat.min()),
            max(f_aimd_flat.max(), f_bvff_flat.max()),
        ]
        ax_f.plot(lim_f, lim_f, "r--", lw=1.2, label="ideal")
        rmse_f = compute_rmse(f_bvff_flat, f_aimd_flat)
        ax_f.set_xlabel("AIMD Force (eV/Å)")
        ax_f.set_ylabel("BVFF Force (eV/Å)")
        ax_f.set_title(f"Force Parity [{label}]\nRMSE = {rmse_f:.4f} eV/Å")
        ax_f.legend(fontsize=8)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "parity_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Parity plot saved to {out / 'parity_plot.png'}.")


# ──────────────────────────────────────────────
# Force distribution
# ──────────────────────────────────────────────

def plot_force_distribution(
    train_data: dict,
    test_data:  dict,
    output_dir: str,
) -> None:
    """
    Plot force magnitude distribution for BVFF vs AIMD.
    Saves to output_dir/force_distribution.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, label, data in zip(axes, ["train", "test"], [train_data, test_data]):
        f_aimd_mag = np.linalg.norm(data["f_aimd"], axis=1)
        f_bvff_mag = np.linalg.norm(data["f_bvff"], axis=1)

        bins = np.linspace(0, max(f_aimd_mag.max(), f_bvff_mag.max()), 50)
        ax.hist(f_aimd_mag, bins=bins, alpha=0.6, label="AIMD",  color="steelblue",  density=True)
        ax.hist(f_bvff_mag, bins=bins, alpha=0.6, label="BVFF",  color="darkorange", density=True)
        ax.set_xlabel("|F| (eV/Å)")
        ax.set_ylabel("Density")
        ax.set_title(f"Force Magnitude Distribution [{label}]")
        ax.legend()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "force_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Force distribution plot saved to {out / 'force_distribution.png'}.")


# ──────────────────────────────────────────────
# Main analysis runner
# ──────────────────────────────────────────────

def run_analysis(
    bvff:        BVFF,
    train_frames: list[Frame],
    test_frames:  list[Frame],
    output_dir:   str,
) -> dict:
    """
    Run full analysis: RMSE summary + parity plots + force distribution.

    Args:
        bvff:         fitted BVFF potential
        train_frames: training frames
        test_frames:  test frames
        output_dir:   directory to save plots

    Returns:
        dict with RMSE values
    """
    logger.info("Running analysis ...")

    train_data = collect_predictions(bvff, train_frames, label="train")
    test_data  = collect_predictions(bvff, test_frames,  label="test")

    # Align energies by the constant offset the fit ignores — and do it PER
    # ref_group, matching the loss. A classical FF energy is defined only up to
    # an additive constant, and frames from different DFT references (e.g. AIMD
    # Γ-only vs DFT 4×4×4) have *different* such constants, so one global offset
    # would distort a mixed-reference parity. The per-group offset c_g is fit on
    # train and applied to test (honest held-out check); test-only groups are
    # warned and left at the global offset.
    tr_groups = np.array([getattr(f, "ref_group", "default") for f in train_frames])
    te_groups = np.array([getattr(f, "ref_group", "default") for f in test_frames])
    global_off = float(np.mean(train_data["e_bvff"] - train_data["e_aimd"]))
    offsets = {}
    for g in np.unique(tr_groups):
        m = tr_groups == g
        offsets[g] = float(np.mean(train_data["e_bvff"][m] - train_data["e_aimd"][m]))
    train_data["e_bvff"] = train_data["e_bvff"] - np.array([offsets[g] for g in tr_groups])
    te_off = []
    for g in te_groups:
        if g not in offsets:
            logger.warning(f"  test-only ref_group '{g}'; using global offset for it.")
        te_off.append(offsets.get(g, global_off))
    test_data["e_bvff"] = test_data["e_bvff"] - np.array(te_off) if len(te_groups) else test_data["e_bvff"]
    if len(offsets) == 1:
        logger.info(f"  Energy reference offset removed: {global_off:.4f} eV/atom (fit up to a constant).")
    else:
        logger.info(f"  Per-ref_group energy offsets removed (eV/atom): "
                    f"{ {g: round(o, 4) for g, o in offsets.items()} }")

    rmse = print_rmse_summary(train_data, test_data, logger)
    logger.info("Generating parity plot ...")
    plot_parity(train_data, test_data, output_dir)
    logger.info("Generating force distribution plot ...")
    plot_force_distribution(train_data, test_data, output_dir)

    # Save RMSE to file
    out = Path(output_dir)
    with open(out / "rmse_summary.txt", "w") as f:
        f.write("RMSE Summary\n")
        f.write("=" * 40 + "\n")
        for k, v in rmse.items():
            f.write(f"{k:20s}: {v:.6f}\n")

    logger.info(f"RMSE summary saved to {out / 'rmse_summary.txt'}.")

    return rmse
