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
    # How the train/test cut is made within each source file:
    #   "random" — shuffle then cut (iid within each temperature). CAVEAT: AIMD
    #              frames a few strides apart are strongly time-correlated, so
    #              every test frame has near-duplicates in train and train≈test
    #              RMSE is guaranteed by construction — an optimizer sanity
    #              check, NOT an overfitting test.
    #   "block"  — hold out the temporally LAST (1 - train_ratio) contiguous
    #              fraction of each trajectory. Test frames are decorrelated
    #              from train; this is the honest generalization metric.
    "split_mode":  "random",

    "potentials": {
        "use_coulomb":    1,
        "use_repulsive":  1,
        "use_buckingham": 0,        # Born-Mayer + dispersion pair term
        "use_BV":         1,
        "use_BVV":        1,
        "use_angle":      0,
        "bv_form":        "exp",    # bond-valence form: "power" | "exp" (Brown-Altermatt)
    },

    "extensions": {
        "use_ewald": 1,
        # Ewald surface (dipole) term boundary: inf = tinfoil (term vanishes;
        # standard for bulk crystals), 1.0 = vacuum. Vacuum penalizes uniformly
        # polarized cells — it suppresses exactly the ferroelectric double-well
        # states this code is meant to fit — and jumps when an atom wraps
        # across the cell boundary, so use it only for genuinely finite systems.
        "ewald_epsilon": float("inf"),
        # Uniform external electric field [Ex, Ey, Ez] in V/Å acting on the
        # rigid-ion charges (f_i = q_i E). ANALYSIS-only knob for polarization
        # switching / hysteresis MD via the calculator — leave zero when
        # fitting (field-free DFT data + a field would bias every parameter;
        # main.py warns if a fit runs with a nonzero field).
        "efield": [0.0, 0.0, 0.0],
    },

    "fitting": {
        "w_E":         1.0,
        "w_F":         1.0,
        "w_S":         1.0,
        # Include the virial stress term in the loss. Off by default: it needs
        # stress in the dataset (and is only used by optimizer="anneal"; the
        # least_squares path fits energy+forces). Every term now has an
        # analytic virial, so the cost is comparable to a force evaluation.
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
        # Fit charges under a cell charge-neutrality constraint: all charges
        # but the most abundant species are fit freely, and that species'
        # charge is fixed by Σ q_i n_i = 0. Required for physical Ewald/Coulomb
        # energies. Set 0 to fit every charge independently (rarely wanted).
        "charge_neutral": 1,
        # Fit the Coulomb charges (1) or hold them fixed at the parameters.toml
        # values (0, recommended). Fitting charges against energies+forces alone
        # is under-determined for a bond-valence FF — the Coulomb term is
        # degenerate with the repulsion/BV terms, so a free fit collapses the
        # charges toward zero (no ferroelectric driving force). Keep them fixed at
        # physical formal/Born values (as the canonical Rappe BVFF does).
        "fit_charges": 0,
        # joblib process budget. 1 = serial (default, unchanged behavior); -1 =
        # all available cores. The budget is spent on a single parallel level so
        # cores are never oversubscribed: the per-frame loss loop (single start),
        # the ensemble of starts (n_starts > 1), or the polish FD gradient.
        "n_jobs":      1,
        # Independent dual_annealing runs forming a multi-start ensemble; the
        # best result is kept. 1 = single start. With n_starts > 1 the runs are
        # parallelized over n_jobs (each run serial-framed), giving better core
        # use and a more robust escape from shallow basins than one long run.
        # (Only used by optimizer="anneal".)
        "n_starts":    1,
        # Optimizer: "least_squares" (default) is a seeded trust-region solver
        # that exploits the physical parameters.toml start — the right tool when a
        # good seed exists. "anneal" is the derivative-free global dual_annealing
        # (+ optional polish), for fitting from scratch with no usable seed.
        "optimizer":   "least_squares",
        # least_squares Jacobian: "analytic" assembles exact columns from each
        # term's parameter gradients (~n_params× fewer E+F passes per iteration
        # than FD, and no FD noise limiting trust-region convergence); "fd" is
        # scipy's 2-point fallback. Charges (fit_charges=1) always use FD
        # columns within the analytic path.
        "jac":         "analytic",
        # Multi-source fitting. weight_normalization="count" makes each frame's
        # per-channel weight = weight_X / N_ref_group (a group-total share, so the
        # static:MD balance is stride-invariant); "none" uses raw per-frame
        # weights. lambda_reg = Tikhonov pull toward the parameters.toml seed
        # (0 = off). Defaults reproduce single-group, flat-weight behavior.
        "weight_normalization": "count",
        "lambda_reg":  0.0,
        # Optimize magnitude parameters (repulsive B, Buckingham A/C, BV S,
        # BVV D) as log10(value): collapse-to-zero becomes a visible walk to
        # the log lower bound instead of a silent 1e-37, and the FD Jacobian
        # is conditioned across the orders of magnitude these prefactors span.
        "log_space":   1,
    },
}


# ──────────────────────────────────────────────
# parameters.toml defaults
# ──────────────────────────────────────────────

DEFAULT_PARAMETERS = {
    "cutoff": 6.0,

    # C² cutoff-taper width in Å (0 disables). Applied to every short-range
    # term (Repulsive, Buckingham, direct Coulomb, BV/BVV via V_ij, Angle) so
    # energy and forces go continuously to zero at the cutoff — without it MD
    # leaks/pumps φ(r_c) at every cutoff crossing and energy is not conserved.
    # Ewald real-space is exempt (erfc is already ≈ 0 at the cutoff by
    # construction of alpha). Lives in parameters.toml (next to cutoff) so a
    # fitted_parameters.toml fully records the model it was fit with.
    "smooth_width": 1.0,

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
