import tomllib
import warnings
from pathlib import Path
from dataclasses import dataclass, field

from .defaults import DEFAULT_CONTROLS
from .dataset import DatasetEntry


@dataclass
class PotentialControls:
    use_coulomb:    bool = True
    use_repulsive:  bool = True
    use_buckingham: bool = False
    use_BV:         bool = True
    use_BVV:        bool = True
    use_angle:      bool = False
    # Must match DEFAULT_CONTROLS["potentials"]["bv_form"]: parse_controls
    # reads from DEFAULT_CONTROLS, but tests/scripts constructing
    # PotentialControls() directly get THIS default — they drifted apart once
    # ("power" here vs "exp" there), silently switching the BV form.
    bv_form:        str  = "exp"     # bond-valence form: "power" | "exp"


@dataclass
class ExtensionControls:
    use_ewald: bool = True
    # Dielectric constant of the medium surrounding the periodic images in the
    # Ewald surface (dipole) term. inf = tinfoil boundary (term vanishes) — the
    # standard choice for bulk crystals and the default; 1.0 = vacuum boundary,
    # which penalizes uniformly polarized cells (depolarizing field) and is
    # discontinuous when an atom wraps across the cell boundary.
    ewald_epsilon: float = float("inf")
    # Uniform external field [Ex, Ey, Ez] (V/Å) on the rigid-ion charges —
    # analysis-only (switching/hysteresis MD); keep zero for fitting.
    efield: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FittingControls:
    w_E:         float = 1.0
    w_F:         float = 1.0
    w_S:         float = 1.0
    use_stress:  bool  = False # include virial stress in the loss (needs stress data)
    polish:      bool  = True  # L-BFGS-B local refinement after simulated annealing
    target_loss: float = 0.0   # stop when best loss < target_loss (0 disables)
    patience:    int   = 0     # stop after N evaluations w/o improvement (0 disables)
    maxiter:     int   = 1000  # hard cap on dual_annealing outer iterations
    charge_neutral: bool = True # fit charges under a cell-neutrality constraint (Ewald needs neutral cells)
    fit_charges: bool  = False # fit charges (True) or hold them fixed at parameters.toml values (False, recommended)
    n_jobs:      int   = 1     # joblib process budget; 1=serial, -1=all cores
    n_starts:    int   = 1     # independent SA runs (multi-start ensemble); best is kept
    optimizer:   str   = "least_squares"  # "least_squares" (seeded trust-region) | "anneal" (global)
    jac:         str   = "analytic"       # least_squares Jacobian: "analytic" (param_grads) | "fd" (2-point)
    weight_normalization: str = "count"   # "count" (per-frame weight = weight_X/N_ref_group) | "none" (raw)
    lambda_reg:  float = 0.0   # Tikhonov pull toward the parameters.toml seed (0 = off)
    log_space:   bool  = True  # fit magnitude params (repulsive B, Buckingham A/C, BV S, BVV D) as log10(value)


@dataclass
class Controls:
    # Input. Normalized at parse time into list[DatasetEntry]. Accepted TOML
    # forms (see controls.toml for examples):
    #   dataset = "file.xml"                       # single file
    #   dataset = ["a.extxyz", "b.extxyz"]         # list of files
    #   [[dataset]] path = "a" ; frame_start = 100 # per-file window override
    # Format is auto-detected from extension
    # (.xml → vasprun, .xyz/.extxyz → extxyz, .parquet/.pq → parquet).
    dataset:     list[DatasetEntry] = field(default_factory=list)

    # Output
    output_dir:  str   = DEFAULT_CONTROLS["output_dir"]
    log_file:    str   = DEFAULT_CONTROLS["log_file"]

    # Task
    task:        str   = DEFAULT_CONTROLS["task"]

    # Train/test split
    train_ratio: float = DEFAULT_CONTROLS["train_ratio"]
    split_seed:  int   = DEFAULT_CONTROLS["split_seed"]   # seed for per-temperature shuffle
    # "random" (shuffle then cut — iid but time-correlated with train) or
    # "block" (hold out the temporally last fraction — the honest metric).
    split_mode:  str   = DEFAULT_CONTROLS["split_mode"]

    # Potential & extension toggles
    potentials:  PotentialControls = field(default_factory=PotentialControls)
    extensions:  ExtensionControls = field(default_factory=ExtensionControls)

    # Fitting weights
    fitting:     FittingControls   = field(default_factory=FittingControls)


def _parse_potentials(data: dict) -> PotentialControls:
    p = data.get("potentials", {})
    d = DEFAULT_CONTROLS["potentials"]
    return PotentialControls(
        use_coulomb    = bool(p.get("use_coulomb",    d["use_coulomb"])),
        use_repulsive  = bool(p.get("use_repulsive",  d["use_repulsive"])),
        use_buckingham = bool(p.get("use_buckingham", d["use_buckingham"])),
        use_BV         = bool(p.get("use_BV",         d["use_BV"])),
        use_BVV        = bool(p.get("use_BVV",        d["use_BVV"])),
        use_angle      = bool(p.get("use_angle",      d["use_angle"])),
        bv_form        = str (p.get("bv_form",        d["bv_form"])),
    )


def _parse_extensions(data: dict) -> ExtensionControls:
    e = data.get("extensions", {})
    d = DEFAULT_CONTROLS["extensions"]
    raw_field = e.get("efield", d["efield"])
    try:
        efield = tuple(float(v) for v in raw_field)
        if len(efield) != 3:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"extensions.efield must be a list of 3 numbers [Ex, Ey, Ez] in "
            f"V/Å, got {raw_field!r}."
        )
    return ExtensionControls(
        use_ewald     = bool (e.get("use_ewald",     d["use_ewald"])),
        ewald_epsilon = float(e.get("ewald_epsilon", d["ewald_epsilon"])),
        efield        = efield,
    )


def _norm_split(value, where: str):
    """
    Normalize a dataset ``split`` value. Accepted:
      true            → frame participates in the ratio split (default)
      false / "train" → pinned entirely to train (e.g. a sparse DFT anchor)
      "test"          → pinned entirely to test (hold out a whole file/condition,
                        e.g. leave-one-temperature-out validation)
    Internally: True (ratio), False (train), "test" (test).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)                    # 1/0 style, like the use_* toggles
    if isinstance(value, str):
        s = value.strip().lower()
        if s == "train":
            return False
        if s == "test":
            return "test"
    raise ValueError(
        f"split must be true, false, 'train', or 'test', got {value!r} ({where})."
    )


def _parse_dataset(data: dict) -> list[DatasetEntry]:
    """
    Normalize the `dataset` field into a list of DatasetEntry. Top-level
    frame_start / frame_end / stride act as defaults; per-entry tables may
    override any of them.
    """
    d = DEFAULT_CONTROLS
    top_start  = int(data.get("frame_start", d["frame_start"]))
    top_end    = int(data.get("frame_end",   d["frame_end"]))
    top_stride = int(data.get("stride",      d["stride"]))
    # Top-level fit-metadata defaults (per-entry tables may override), mirroring
    # how frame_start works. Default ref_group="default" + unit weights reproduce
    # single-group, flat-weight behavior.
    top_group  = str  (data.get("ref_group", "default"))
    top_wE     = float(data.get("weight_E",  1.0))
    top_wF     = float(data.get("weight_F",  1.0))
    top_wS     = float(data.get("weight_S",  1.0))
    top_split  = _norm_split(data.get("split", True), "top-level")

    raw = data.get("dataset", d["dataset"])
    items = [raw] if isinstance(raw, (str, dict)) else list(raw)
    if not items:
        raise ValueError("dataset must specify at least one file.")

    entries: list[DatasetEntry] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            entries.append(DatasetEntry(
                path        = item,
                frame_start = top_start,
                frame_end   = top_end,
                stride      = top_stride,
                ref_group   = top_group,
                weight_E    = top_wE,
                weight_F    = top_wF,
                weight_S    = top_wS,
                split       = top_split,
            ))
        elif isinstance(item, dict):
            if "path" not in item:
                raise ValueError(f"dataset entry #{i} is missing required 'path' key: {item}")
            entries.append(DatasetEntry(
                path        = str(item["path"]),
                frame_start = int(item.get("frame_start", top_start)),
                frame_end   = int(item.get("frame_end",   top_end)),
                stride      = int(item.get("stride",      top_stride)),
                ref_group   = str  (item.get("ref_group", top_group)),
                weight_E    = float(item.get("weight_E",  top_wE)),
                weight_F    = float(item.get("weight_F",  top_wF)),
                weight_S    = float(item.get("weight_S",  top_wS)),
                split       = _norm_split(item.get("split", top_split),
                                          f"dataset entry #{i}"),
            ))
        else:
            raise ValueError(
                f"dataset entry #{i} must be a string or table, got {type(item).__name__}."
            )
    return entries


def _parse_fitting(data: dict) -> FittingControls:
    f = data.get("fitting", {})
    d = DEFAULT_CONTROLS["fitting"]
    return FittingControls(
        w_E         = float(f.get("w_E",         d["w_E"])),
        w_F         = float(f.get("w_F",         d["w_F"])),
        w_S         = float(f.get("w_S",         d["w_S"])),
        use_stress  = bool (f.get("use_stress",  d["use_stress"])),
        polish      = bool (f.get("polish",      d["polish"])),
        target_loss = float(f.get("target_loss", d["target_loss"])),
        patience    = int  (f.get("patience",    d["patience"])),
        maxiter     = int  (f.get("maxiter",     d["maxiter"])),
        charge_neutral = bool(f.get("charge_neutral", d["charge_neutral"])),
        fit_charges = bool (f.get("fit_charges",  d["fit_charges"])),
        n_jobs      = int  (f.get("n_jobs",      d["n_jobs"])),
        n_starts    = int  (f.get("n_starts",    d["n_starts"])),
        optimizer   = str  (f.get("optimizer",   d["optimizer"])),
        jac         = str  (f.get("jac",         d["jac"])),
        weight_normalization = str(f.get("weight_normalization", d["weight_normalization"])),
        lambda_reg  = float(f.get("lambda_reg",  d["lambda_reg"])),
        log_space   = bool (f.get("log_space",   d["log_space"])),
    )


def _validate(controls: Controls) -> None:
    valid_tasks = {"energy", "force", "both"}
    if controls.task not in valid_tasks:
        raise ValueError(f"Invalid task '{controls.task}'. Must be one of {valid_tasks}.")

    if controls.potentials.bv_form not in ("power", "exp"):
        raise ValueError(
            f"potentials.bv_form must be 'power' or 'exp', got '{controls.potentials.bv_form}'."
        )

    # Potential-term sanity: at least one term must be active, and some
    # combinations silently shadow or double-count each other. Warn rather
    # than error for the latter — they are valid, just rarely intended.
    pc = controls.potentials
    if not any([pc.use_coulomb, pc.use_repulsive, pc.use_buckingham,
                pc.use_BV, pc.use_BVV, pc.use_angle]):
        raise ValueError(
            "No potential terms enabled in [potentials]. Enable at least one "
            "of use_coulomb / use_repulsive / use_buckingham / use_BV / use_BVV / use_angle."
        )
    if pc.use_repulsive and pc.use_buckingham:
        warnings.warn(
            "Both use_repulsive and use_buckingham are enabled: the short-range "
            "repulsion is counted twice (r^-12 AND Born-Mayer). Usually only one "
            "is intended.", stacklevel=2,
        )
    if pc.use_coulomb and controls.extensions.use_ewald:
        warnings.warn(
            "use_coulomb and extensions.use_ewald are both enabled: the direct "
            "Coulomb sum is ignored — Ewald replaces it. Set use_ewald=0 to use "
            "the direct sum.", stacklevel=2,
        )
    if controls.extensions.ewald_epsilon < 1.0:
        raise ValueError(
            f"extensions.ewald_epsilon must be >= 1 (1.0 = vacuum, inf = tinfoil), "
            f"got {controls.extensions.ewald_epsilon}."
        )

    if not (0.0 < controls.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {controls.train_ratio}.")

    if controls.split_mode not in ("random", "block"):
        raise ValueError(
            f"split_mode must be 'random' or 'block', got '{controls.split_mode}'."
        )

    if not controls.dataset:
        raise ValueError("dataset must specify at least one file.")

    for e in controls.dataset:
        if e.frame_start < 0:
            raise ValueError(
                f"frame_start must be >= 0, got {e.frame_start} (file '{e.path}')."
            )
        if e.frame_end != -1 and e.frame_end <= e.frame_start:
            raise ValueError(
                f"frame_end must be greater than frame_start or -1 (file '{e.path}')."
            )
        if e.stride < 1:
            raise ValueError(
                f"stride must be >= 1, got {e.stride} (file '{e.path}')."
            )
        if not Path(e.path).exists():
            raise FileNotFoundError(f"dataset file not found: '{e.path}'.")

    for w_name, w_val in [("w_E", controls.fitting.w_E), ("w_F", controls.fitting.w_F), ("w_S", controls.fitting.w_S)]:
        if w_val < 0:
            raise ValueError(f"Fitting weight {w_name} must be >= 0, got {w_val}.")

    if controls.fitting.target_loss < 0:
        raise ValueError(f"fitting.target_loss must be >= 0, got {controls.fitting.target_loss}.")
    if controls.fitting.patience < 0:
        raise ValueError(f"fitting.patience must be >= 0, got {controls.fitting.patience}.")
    if controls.fitting.maxiter <= 0:
        raise ValueError(f"fitting.maxiter must be > 0, got {controls.fitting.maxiter}.")
    if controls.fitting.n_jobs == 0:
        raise ValueError(f"fitting.n_jobs must be != 0 (1=serial, -1=all cores), got {controls.fitting.n_jobs}.")
    if controls.fitting.n_starts < 1:
        raise ValueError(f"fitting.n_starts must be >= 1, got {controls.fitting.n_starts}.")
    if controls.fitting.optimizer not in ("least_squares", "anneal"):
        raise ValueError(
            f"fitting.optimizer must be 'least_squares' or 'anneal', got "
            f"'{controls.fitting.optimizer}'."
        )
    if controls.fitting.jac not in ("analytic", "fd"):
        raise ValueError(
            f"fitting.jac must be 'analytic' or 'fd', got '{controls.fitting.jac}'."
        )
    if controls.fitting.weight_normalization not in ("count", "none"):
        raise ValueError(
            f"fitting.weight_normalization must be 'count' or 'none', got "
            f"'{controls.fitting.weight_normalization}'."
        )
    if controls.fitting.lambda_reg < 0:
        raise ValueError(f"fitting.lambda_reg must be >= 0, got {controls.fitting.lambda_reg}.")
    for e in controls.dataset:
        for wn, wv in [("weight_E", e.weight_E), ("weight_F", e.weight_F), ("weight_S", e.weight_S)]:
            if wv < 0:
                raise ValueError(f"dataset {wn} must be >= 0, got {wv} (file '{e.path}').")


def parse_controls(filepath: str = "controls.toml", validate: bool = True) -> Controls:
    """
    Parse controls.toml and return a Controls dataclass.

    Args:
        filepath: Path to controls.toml
        validate: If True, validate the parsed values

    Returns:
        Controls dataclass
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"controls.toml not found: '{filepath}'.")

    with open(path, "rb") as f:
        data = tomllib.load(f)

    d = DEFAULT_CONTROLS
    controls = Controls(
        dataset     = _parse_dataset(data),
        output_dir  = data.get("output_dir",  d["output_dir"]),
        log_file    = data.get("log_file",    d["log_file"]),
        task        = data.get("task",        d["task"]),
        train_ratio = data.get("train_ratio", d["train_ratio"]),
        split_seed  = int(data.get("split_seed", d["split_seed"])),
        split_mode  = str(data.get("split_mode", d["split_mode"])),
        potentials  = _parse_potentials(data),
        extensions  = _parse_extensions(data),
        fitting     = _parse_fitting(data),
    )

    if validate:
        _validate(controls)

    return controls


if __name__ == "__main__":
    controls = parse_controls("controls.toml", validate=False)
    print(controls)
