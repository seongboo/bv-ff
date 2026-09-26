import tomllib
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
    bv_form:        str  = "power"   # bond-valence form: "power" | "exp"


@dataclass
class ExtensionControls:
    use_ewald:      bool  = True
    ewald_cutoff:   float = 8.0    # Ewald real-space cutoff r_c (Å), independent of short-range cutoff
    ewald_accuracy: float = 1e-6   # target truncation accuracy δ → α = √(-ln δ)/r_c, k_c = 2α√(-ln δ)


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


@dataclass
class Controls:
    # Input. Normalized at parse time into list[DatasetEntry]. Accepted TOML
    # forms (see controls.toml for examples):
    #   dataset = "file.xml"                       # single file
    #   dataset = ["a.extxyz", "b.extxyz"]         # list of files
    #   [[dataset]] path = "a" ; frame_start = 100 # per-file window override
    # Format is auto-detected from extension (.xml → vasprun, .xyz/.extxyz → extxyz).
    dataset:     list[DatasetEntry] = field(default_factory=list)

    # Output
    output_dir:  str   = DEFAULT_CONTROLS["output_dir"]
    log_file:    str   = DEFAULT_CONTROLS["log_file"]

    # Task
    task:        str   = DEFAULT_CONTROLS["task"]

    # Train/test split
    train_ratio: float = DEFAULT_CONTROLS["train_ratio"]
    split_seed:  int   = DEFAULT_CONTROLS["split_seed"]   # seed for per-temperature shuffle

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
    return ExtensionControls(
        use_ewald      = bool (e.get("use_ewald",      d["use_ewald"])),
        ewald_cutoff   = float(e.get("ewald_cutoff",   d["ewald_cutoff"])),
        ewald_accuracy = float(e.get("ewald_accuracy", d["ewald_accuracy"])),
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
            ))
        elif isinstance(item, dict):
            if "path" not in item:
                raise ValueError(f"dataset entry #{i} is missing required 'path' key: {item}")
            entries.append(DatasetEntry(
                path        = str(item["path"]),
                frame_start = int(item.get("frame_start", top_start)),
                frame_end   = int(item.get("frame_end",   top_end)),
                stride      = int(item.get("stride",      top_stride)),
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
    )


def _validate(controls: Controls) -> None:
    valid_tasks = {"energy", "force", "both"}
    if controls.task not in valid_tasks:
        raise ValueError(f"Invalid task '{controls.task}'. Must be one of {valid_tasks}.")

    if controls.potentials.bv_form not in ("power", "exp"):
        raise ValueError(
            f"potentials.bv_form must be 'power' or 'exp', got '{controls.potentials.bv_form}'."
        )

    if controls.extensions.ewald_cutoff <= 0:
        raise ValueError(
            f"extensions.ewald_cutoff must be > 0, got {controls.extensions.ewald_cutoff}."
        )
    if not (0.0 < controls.extensions.ewald_accuracy < 1.0):
        raise ValueError(
            f"extensions.ewald_accuracy must be in (0, 1), got {controls.extensions.ewald_accuracy}."
        )

    if not (0.0 < controls.train_ratio < 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {controls.train_ratio}.")

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
