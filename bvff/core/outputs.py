from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import numpy as np

from .potentials import BVFF


T = TypeVar("T")


# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────

# Minimum spacing of periodic progress lines (optimizer iterations, MD steps).
# Time-based rather than every-N-iterations, since one iteration costs anywhere
# from milliseconds to minutes depending on the dataset size.
LOG_INTERVAL_S = 10.0


class LogThrottle:
    """Rate limiter for periodic progress lines: ``ready()`` is True on the
    first call, then at most once per ``interval`` seconds (``force=True``
    always passes and restarts the interval)."""

    def __init__(self, interval: float = LOG_INTERVAL_S):
        self.interval = interval
        self._last: float | None = None

    def ready(self, force: bool = False) -> bool:
        now = time.monotonic()
        if force or self._last is None or now - self._last >= self.interval:
            self._last = now
            return True
        return False


def init_cli_output() -> None:
    """Line-buffer stdout/stderr so output appears immediately even when piped
    or redirected (SLURM, ``| tee``) — Python block-buffers a non-TTY stdout.
    Call once at the top of every command-line entry point."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass


def cli_logger() -> logging.Logger:
    """The 'bvff' logger for a standalone tool: stdout only, same format as a
    fit log. Leaves an already-configured logger (e.g. inside bvff-fit) as is."""
    init_cli_output()
    logger = logging.getLogger("bvff")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                          datefmt="%H:%M:%S"))
        logger.addHandler(ch)
    return logger


def setup_logger(log_file: str, output_dir: str) -> logging.Logger:
    """Configure the 'bvff' logger with both file and unbuffered stdout output."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    init_cli_output()

    logger = logging.getLogger("bvff")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(Path(output_dir) / log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def section(logger: logging.Logger, title: str) -> None:
    """Emit a visually distinct section banner."""
    logger.info("─" * 60)
    logger.info(title)
    logger.info("─" * 60)


# ──────────────────────────────────────────────
# Progress bar
# ──────────────────────────────────────────────

def _format_bar(frac: float, width: int) -> str:
    frac   = max(0.0, min(1.0, frac))
    filled = int(round(width * frac))
    return "█" * filled + "░" * (width - filled)


def _draw_bar(stream, label: str, i: int, total: int, width: int, elapsed: float) -> None:
    frac = i / total if total else 1.0
    eta  = (elapsed / frac - elapsed) if frac > 0 else 0.0
    stream.write(
        f"\r{label}: {frac * 100:5.1f}% [{_format_bar(frac, width)}] "
        f"{i}/{total} | {elapsed:5.1f}s | ETA {eta:5.1f}s"
    )
    stream.flush()


def progress_iter(
    items:        Iterable[T],
    label:        str = "Progress",
    total:        int | None = None,
    width:        int = 50,
    min_interval: float = 0.1,
    stream       = None,
) -> Iterator[T]:
    """
    Wrap an iterable with an in-place terminal progress bar.

    On a TTY, redraws a single line via '\\r' at most every ``min_interval``
    seconds. On a non-TTY (output piped to a file), prints a plain line every
    ~2 s so progress is still visible in the captured log.
    """
    if stream is None:
        stream = sys.stdout
    if total is None:
        try:
            total = len(items)  # type: ignore[arg-type]
        except TypeError:
            total = 0

    is_tty = getattr(stream, "isatty", lambda: False)()
    t0     = time.time()
    last   = 0.0

    if is_tty and total:
        _draw_bar(stream, label, 0, total, width, 0.0)

    for i, item in enumerate(items, start=1):
        yield item

        now = time.time()
        if total <= 0:
            continue

        if is_tty:
            if i < total and now - last < min_interval:
                continue
            last = now
            _draw_bar(stream, label, i, total, width, now - t0)
        else:
            if i == 1 or i == total or now - last >= 2.0:
                pct = 100.0 * i / total
                print(
                    f"{label}: {pct:5.1f}% ({i}/{total}) | elapsed {now - t0:.1f}s",
                    file=stream,
                    flush=True,
                )
                last = now

    if is_tty and total:
        stream.write("\n")
        stream.flush()


# ──────────────────────────────────────────────
# Progress-aware prediction
# ──────────────────────────────────────────────

def _stack_forces(forces: list[np.ndarray]) -> np.ndarray:
    """Stack per-frame (N_i, 3) force arrays. Homogeneous atom counts stack to
    a regular (F, N, 3) array; a mixed-cell dataset (e.g. a 5-atom EOS anchor
    plus a 40-atom AIMD supercell) is ragged, where a plain ``np.array`` call
    raises "inhomogeneous shape" — return a 1-D object array of per-frame
    arrays instead (readable via ``np.load(..., allow_pickle=True)``)."""
    if len({f.shape for f in forces}) <= 1:
        return np.array(forces)
    out = np.empty(len(forces), dtype=object)
    for i, f in enumerate(forces):
        out[i] = f
    return out


def predict_with_progress(
    bvff:    BVFF,
    frames,
    kind:    str,          # "energy", "force", or "both"
    label:   str,          # "train" / "test"
    logger:  logging.Logger,
):
    """Compute predictions frame-by-frame with a progress bar.

    ``kind="both"`` returns ``(energies, forces)`` from a single
    ``energy_and_forces`` pass per frame — half the cost of two separate
    energy-then-forces sweeps, since the terms share their intermediates.
    """
    if len(frames) == 0:
        empty = np.array([])
        return (empty, empty) if kind == "both" else empty

    t0 = time.time()
    energies, forces = [], []
    for f in progress_iter(frames, label=f"{label}/{kind:6s}"):
        if kind == "energy":
            energies.append(bvff.energy(f.lattice, f.species, f.positions))
        elif kind == "force":
            forces.append(bvff.forces(f.lattice, f.species, f.positions))
        else:
            e, frc = bvff.energy_and_forces(f.lattice, f.species, f.positions)
            energies.append(e)
            forces.append(frc)

    logger.info(f"  [{label}/{kind}] done in {time.time() - t0:.1f}s")
    if kind == "energy":
        return np.array(energies)
    if kind == "force":
        return _stack_forces(forces)
    return np.array(energies), _stack_forces(forces)


# ──────────────────────────────────────────────
# Save predictions
# ──────────────────────────────────────────────

def _sha256_file(path: str | Path, n_hex: int = 16) -> str:
    """Short sha256 of a file's contents (16 hex chars ≈ 64 bits — plenty to
    detect that an input changed between two runs)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n_hex]


def _toml_safe(obj):
    """Recursively convert a value into something tomli_w accepts: dataclasses
    → dicts, tuples → lists, numpy scalars → Python scalars, None dropped
    (TOML has no null)."""
    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _toml_safe(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_toml_safe(v) for v in obj if v is not None]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_provenance(
    output_dir:      str,
    controls,
    train_frames,
    test_frames,
    fit_diagnostics: dict,
    params_file:     str = "parameters.toml",
    elapsed_s:       float | None = None,
    logger:          logging.Logger | None = None,
) -> None:
    """
    Persist ``fit_provenance.toml`` next to fitted_parameters.toml: everything
    needed to answer, months later, "what EXACTLY produced this fit?" —
    inputs (dataset files + hashes + split), configuration (full resolved
    controls snapshot), environment (package versions), code state (per-file
    source hashes; this repo has no git history, so the hashes ARE the
    version), and the fit outcome (losses, optimizer, bound flags).
    """
    import platform
    from collections import Counter
    import tomli_w

    root = Path(__file__).resolve().parents[1]        # the bvff/ package

    src_hashes = {}
    for sub in ("core", "core/extensions", "parsers", "tools"):
        d = root / sub
        if d.is_dir():
            for f in sorted(d.glob("*.py")):
                src_hashes[f"bvff/{sub}/{f.name}"] = _sha256_file(f)
    combined = hashlib.sha256(
        "".join(f"{k}{v}" for k, v in sorted(src_hashes.items())).encode()
    ).hexdigest()[:16]

    def _versions():
        out = {"python": platform.python_version()}
        for mod in ("numpy", "scipy", "ase", "pandas"):
            try:
                out[mod] = __import__(mod).__version__
            except Exception:
                pass
        return out

    tr_src = Counter(f.source for f in train_frames)
    te_src = Counter(f.source for f in test_frames)
    dataset = []
    for e in controls.dataset:
        entry = _toml_safe(e)
        try:
            entry["sha256"] = _sha256_file(e.path)
        except OSError:
            entry["sha256"] = "unreadable"
        entry["n_train"] = int(tr_src.get(e.path, 0))
        entry["n_test"]  = int(te_src.get(e.path, 0))
        dataset.append(entry)

    seed_file = {"path": params_file}
    if Path(params_file).exists():
        seed_file["sha256"] = _sha256_file(params_file)

    doc = {
        "generated": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "hostname":  platform.node(),
            **({"elapsed_s": round(float(elapsed_s), 1)} if elapsed_s is not None else {}),
        },
        "environment": _versions(),
        "source": {"combined_sha256": combined, "files": src_hashes},
        "controls": _toml_safe({
            k: v for k, v in vars(controls).items() if k != "dataset"
        }),
        "dataset": dataset,
        "parameters_seed": seed_file,
        "split": {
            "n_train": len(train_frames),
            "n_test":  len(test_frames),
            "mode":    controls.split_mode,
            "seed":    controls.split_seed,
        },
        "fit": _toml_safe(fit_diagnostics),
    }

    path = Path(output_dir) / "fit_provenance.toml"
    with open(path, "wb") as f:
        tomli_w.dump(_toml_safe(doc), f)
    if logger:
        logger.info(f"Provenance saved to {path} (source state {combined}).")


def _save_forces(path: Path, forces: np.ndarray, logger: logging.Logger) -> None:
    """np.save that tolerates the ragged (object-dtype) mixed-cell case, which
    requires pickling; regular arrays are saved pickle-free as before."""
    if forces.dtype == object:
        np.save(path, forces, allow_pickle=True)
        logger.info(
            f"  {path.name}: mixed atom counts — saved as an object array of "
            f"per-frame (N, 3) arrays; load with np.load(..., allow_pickle=True)."
        )
    else:
        np.save(path, forces)


def save_results(
    bvff:          BVFF,
    output_dir:    str,
    task:          str,
    train_frames,
    test_frames,
    logger:        logging.Logger,
) -> None:
    # controls_parser validates the task, but guard direct callers too: the
    # silent alternative is a run that "finishes" having saved nothing.
    if task not in ("both", "energy", "force"):
        raise ValueError(f"Unknown task '{task}': must be 'both', 'energy', or 'force'.")

    out = Path(output_dir)

    if task == "both":
        logger.info(f"Computing energies+forces (train={len(train_frames)}, test={len(test_frames)}) ...")
        train_energies, train_forces = predict_with_progress(bvff, train_frames, "both", "train", logger)
        test_energies,  test_forces  = predict_with_progress(bvff, test_frames,  "both", "test",  logger)
        np.save(out / "train_energies.npy", train_energies)
        np.save(out / "test_energies.npy",  test_energies)
        _save_forces(out / "train_forces.npy", train_forces, logger)
        _save_forces(out / "test_forces.npy",  test_forces,  logger)
        logger.info(f"Energies and forces saved to {out}.")
        return

    if task == "energy":
        logger.info(f"Computing energies (train={len(train_frames)}, test={len(test_frames)}) ...")
        train_energies = predict_with_progress(bvff, train_frames, "energy", "train", logger)
        test_energies  = predict_with_progress(bvff, test_frames,  "energy", "test",  logger)
        np.save(out / "train_energies.npy", train_energies)
        np.save(out / "test_energies.npy",  test_energies)
        logger.info(f"Energies saved to {out}.")

    if task == "force":
        logger.info(f"Computing forces (train={len(train_frames)}, test={len(test_frames)}) ...")
        train_forces = predict_with_progress(bvff, train_frames, "force", "train", logger)
        test_forces  = predict_with_progress(bvff, test_frames,  "force", "test",  logger)
        _save_forces(out / "train_forces.npy", train_forces, logger)
        _save_forces(out / "test_forces.npy",  test_forces,  logger)
        logger.info(f"Forces saved to {out}.")
