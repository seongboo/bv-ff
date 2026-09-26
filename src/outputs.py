from __future__ import annotations

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

def setup_logger(log_file: str, output_dir: str) -> logging.Logger:
    """Configure the 'bvff' logger with both file and unbuffered stdout output."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Force line-buffered stdout so log lines appear immediately even when piped.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

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

def predict_with_progress(
    bvff:    BVFF,
    frames,
    kind:    str,          # "energy" or "force"
    label:   str,          # "train" / "test"
    logger:  logging.Logger,
) -> np.ndarray:
    """Compute energies or forces frame-by-frame with a progress bar."""
    if len(frames) == 0:
        return np.array([])

    t0 = time.time()
    out = []
    for f in progress_iter(frames, label=f"{label}/{kind:6s}"):
        if kind == "energy":
            out.append(bvff.energy(f.lattice, f.species, f.positions))
        else:
            out.append(bvff.forces(f.lattice, f.species, f.positions))

    logger.info(f"  [{label}/{kind}] done in {time.time() - t0:.1f}s")
    return np.array(out)


# ──────────────────────────────────────────────
# Save predictions
# ──────────────────────────────────────────────

def save_results(
    bvff:          BVFF,
    output_dir:    str,
    task:          str,
    train_frames,
    test_frames,
    logger:        logging.Logger,
) -> None:
    out = Path(output_dir)

    if task in ("energy", "both"):
        logger.info(f"Computing energies (train={len(train_frames)}, test={len(test_frames)}) ...")
        train_energies = predict_with_progress(bvff, train_frames, "energy", "train", logger)
        test_energies  = predict_with_progress(bvff, test_frames,  "energy", "test",  logger)
        np.save(out / "train_energies.npy", train_energies)
        np.save(out / "test_energies.npy",  test_energies)
        logger.info(f"Energies saved to {out}.")

    if task in ("force", "both"):
        logger.info(f"Computing forces (train={len(train_frames)}, test={len(test_frames)}) ...")
        train_forces = predict_with_progress(bvff, train_frames, "force", "train", logger)
        test_forces  = predict_with_progress(bvff, test_frames,  "force", "test",  logger)
        np.save(out / "train_forces.npy", train_forces)
        np.save(out / "test_forces.npy",  test_forces)
        logger.info(f"Forces saved to {out}.")
