# defaults.py
# All default values for controls.toml and parameters.toml


# ──────────────────────────────────────────────
# controls.toml defaults
# ──────────────────────────────────────────────

DEFAULT_CONTROLS = {
    # `dataset` is either a single path or a list of paths. Format is
    # auto-detected from the extension: `.xml` → vasprun, `.xyz`/`.extxyz` → extxyz.
    "dataset":     "vasprun.xml",
    "frame_start": 0,
    "frame_end":   -1,
    "stride":      1,
    "output_dir":  "./output",
    "log_file":    "bvff.log",
    "task":        "both",
    "train_ratio": 0.8,
    # Seed for the per-temperature train/test shuffle. Frames within each
    # source file are shuffled with this seed before splitting, so the test
    # set is iid within each temperature yet fully reproducible.
    "split_seed":  0,

    "potentials": {
        "use_coulomb":    1,
        "use_repulsive":  1,
        "use_buckingham": 0,        # Born-Mayer + dispersion pair term
        "use_BV":         1,
        "use_BVV":        1,
        "use_angle":      0,
        "bv_form":        "power",  # bond-valence form: "power" | "exp"
    },

    "extensions": {
        "use_ewald":      1,
        # Ewald real-space cutoff (Å), separate from the short-range cutoff in
        # parameters.toml, and target accuracy δ. α and k_c follow from these:
        #   α = √(-ln δ)/r_c,  k_c = 2α√(-ln δ)  (tin-foil boundary).
        "ewald_cutoff":   8.0,
        "ewald_accuracy": 1e-6,
    },

    "fitting": {
        "w_E":         1.0,
        "w_F":         1.0,
        "w_S":         1.0,
        # Include the virial stress term in the loss. Off by default: it needs
        # stress in the dataset and adds up to ~18 energy evals/frame for terms
        # without a closed-form virial (Ewald, BV, BVV, Angle).
        "use_stress":  0,
        # Run an L-BFGS-B local refinement on the best point after simulated
        # annealing (cheap polish; keeps whichever result is better).
        "polish":      1,
        # Early stopping (both off by default). target_loss: stop once best
        # loss drops below this. patience: stop after this many evaluations
        # without improvement (0 disables).
        "target_loss": 0.0,
        "patience":    0,
        "maxiter":     1000,
    },
}


# ──────────────────────────────────────────────
# parameters.toml defaults
# ──────────────────────────────────────────────

DEFAULT_PARAMETERS = {
    "cutoff": 6.0,

    "coulomb": {},      # atom: charge

    "repulsive": {},    # pair: B

    "buckingham": {},   # pair: {A, rho, C}

    "BV": {
        "species": {},  # atom: {V0, S}
        "pairs":   {},  # pair: {r0, C}
    },

    "BVV": {},          # atom: {W0, D}

    "angle": {
        "k": 0.0,
    },
}
