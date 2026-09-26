#!/usr/bin/env python3
"""
Generate a `controls.toml` for bvff.

Examples
--------
# Single vasprun
python scripts/gen_controls.py examples/PbTiO3/vasprun.xml

# Multiple extxyz files sharing the same window
python scripts/gen_controls.py data/300K.extxyz data/600K.extxyz \\
    --frame-start 100 --stride 10

# Per-file table-array form (each file gets its own editable window)
python scripts/gen_controls.py data/300K.extxyz data/900K.extxyz --per-file

# Disable a potential, tune fitting
python scripts/gen_controls.py vasprun.xml --no-bvv --task force --w-F 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _toml_str(s: str) -> str:
    """Render a Python string as a TOML basic string."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _fmt_dataset_block(paths: list[str], per_file: bool, win: dict) -> str:
    """
    Render the dataset declaration.

    - If exactly one path and not per_file → `dataset = "path"`.
    - If multiple paths and not per_file   → `dataset = ["a", "b"]`.
    - If per_file                          → `[[dataset]]` table array,
      each entry carrying its own window (seeded with the top-level defaults).
    """
    if not per_file and len(paths) == 1:
        return f"dataset = {_toml_str(paths[0])}\n"

    if not per_file:
        rendered = ", ".join(_toml_str(p) for p in paths)
        return f"dataset = [{rendered}]\n"

    blocks = []
    for p in paths:
        blocks.append(
            "[[dataset]]\n"
            f"path        = {_toml_str(p)}\n"
            f"frame_start = {win['frame_start']}\n"
            f"frame_end   = {win['frame_end']}\n"
            f"stride      = {win['stride']}\n"
        )
    return "\n".join(blocks)


def _b(flag: bool) -> int:
    return 1 if flag else 0


# ──────────────────────────────────────────────
# Template
# ──────────────────────────────────────────────

DATASET_HEADER = """\
# Input dataset. Format auto-detected by extension:
#   *.xml          → VASP vasprun
#   *.xyz/.extxyz  → extxyz (ASE)
#
# Accepted forms:
#   dataset = "file.xml"
#   dataset = ["a.extxyz", "b.extxyz"]
#   [[dataset]] path = "a" ; frame_start = 100 ; stride = 10
"""

WINDOW_HEADER = """\
# Default frame window (used when a [[dataset]] entry doesn't override)
"""


def build_toml(args: argparse.Namespace) -> str:
    """
    TOML constraint: once any `[section]` or `[[section]]` header is opened,
    all subsequent key/value pairs belong to that section. So we lay out the
    file in two phases:
      1. all top-level (root) scalars  — includes the `dataset = ...` form
      2. all section headers           — `[potentials]`, `[extensions]`,
         `[fitting]`, and (if per-file) the `[[dataset]]` entries last.
    """
    paths = [str(p) for p in args.dataset]
    win = dict(
        frame_start = args.frame_start,
        frame_end   = args.frame_end,
        stride      = args.stride,
    )

    out: list[str] = []

    # ── Phase 1: top-level scalars ────────────
    out.append(DATASET_HEADER)
    if args.per_file:
        out.append(
            "# Per-file mode: dataset entries are defined as [[dataset]] "
            "sections at the bottom of this file.\n"
        )
    else:
        out.append(_fmt_dataset_block(paths, per_file=False, win=win))
    out.append("\n")

    out.append(WINDOW_HEADER)
    out.append(f"frame_start = {win['frame_start']}\n")
    out.append(f"frame_end   = {win['frame_end']}\n")
    out.append(f"stride      = {win['stride']}\n")
    out.append("\n")

    out.append("# Output\n")
    out.append(f"output_dir = {_toml_str(args.output_dir)}\n")
    out.append(f"log_file   = {_toml_str(args.log_file)}\n")
    out.append("\n")

    out.append("# Task: energy / force / both\n")
    out.append(f"task = {_toml_str(args.task)}\n")
    out.append("\n")
    out.append("# Train/test split (stratified per source file; frames within each\n")
    out.append("# temperature are shuffled by split_seed before the cut for reproducibility)\n")
    out.append(f"train_ratio = {args.train_ratio}\n")
    out.append(f"split_seed  = {args.split_seed}\n")
    out.append("\n")

    # ── Phase 2: section headers ──────────────
    out.append("[potentials]\n")
    out.append(f"use_coulomb   = {_b(args.coulomb)}\n")
    out.append(f"use_repulsive = {_b(args.repulsive)}\n")
    out.append(f"use_BV        = {_b(args.bv)}\n")
    out.append(f"use_BVV       = {_b(args.bvv)}\n")
    out.append(f"use_angle     = {_b(args.angle)}\n")
    out.append("\n")

    out.append("[extensions]\n")
    out.append(f"use_ewald = {_b(args.ewald)}\n")
    out.append("\n")

    out.append("[fitting]\n")
    out.append(f"w_E         = {args.w_E}\n")
    out.append(f"w_F         = {args.w_F}\n")
    out.append(f"w_S         = {args.w_S}\n")
    out.append("# Early-stop controls (both off by default):\n")
    out.append("#   target_loss > 0 → stop once best loss drops below it\n")
    out.append("#   patience    > 0 → stop after N consecutive evaluations without improvement\n")
    out.append(f"target_loss = {args.target_loss}\n")
    out.append(f"patience    = {args.patience}\n")
    out.append(f"maxiter     = {args.maxiter}\n")

    # Per-file dataset entries come last because they're section headers.
    if args.per_file:
        out.append("\n")
        out.append("# Per-file dataset entries. Edit each block's window independently.\n")
        out.append(_fmt_dataset_block(paths, per_file=True, win=win))

    return "".join(out)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "gen_controls",
        description = "Generate a controls.toml for bvff.",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog      = __doc__,
    )

    p.add_argument(
        "dataset", nargs="*", type=Path,
        help="One or more trajectory files (vasprun.xml or *.extxyz). "
             "Defaults to 'vasprun.xml' if omitted.",
    )
    p.add_argument(
        "-o", "--output", default="controls.toml", type=Path,
        help="Output TOML path (default: controls.toml).",
    )
    p.add_argument(
        "-f", "--force", action="store_true",
        help="Overwrite the output file if it exists.",
    )
    p.add_argument(
        "--per-file", action="store_true",
        help="Emit dataset as a [[dataset]] table array so each file's window "
             "can be edited independently.",
    )
    p.add_argument(
        "--no-check", action="store_true",
        help="Skip the 'do the dataset paths exist?' warning.",
    )

    # Frame window defaults
    g = p.add_argument_group("frame window defaults")
    g.add_argument("--frame-start", type=int, default=0)
    g.add_argument("--frame-end",   type=int, default=-1)
    g.add_argument("--stride",      type=int, default=1)

    # Run config
    g = p.add_argument_group("run")
    g.add_argument("--task", choices=["energy", "force", "both"], default="both")
    g.add_argument("--train-ratio", type=float, default=0.8)
    g.add_argument("--split-seed",  dest="split_seed", type=int, default=0,
                   help="Seed for the per-temperature train/test shuffle (default: 0).")
    g.add_argument("--output-dir", default="./output")
    g.add_argument("--log-file",   default="bvff.log")

    # Potentials (default-on; --no-X to disable)
    g = p.add_argument_group("potentials (default on except --angle)")
    g.add_argument("--no-coulomb",   dest="coulomb",   action="store_false")
    g.add_argument("--no-repulsive", dest="repulsive", action="store_false")
    g.add_argument("--no-bv",        dest="bv",        action="store_false")
    g.add_argument("--no-bvv",       dest="bvv",       action="store_false")
    g.add_argument("--angle",        dest="angle",     action="store_true")
    p.set_defaults(coulomb=True, repulsive=True, bv=True, bvv=True, angle=False)

    # Extensions
    g = p.add_argument_group("extensions")
    g.add_argument("--no-ewald", dest="ewald", action="store_false")
    p.set_defaults(ewald=True)

    # Fitting
    g = p.add_argument_group("fitting")
    g.add_argument("--w-E", dest="w_E", type=float, default=1.0)
    g.add_argument("--w-F", dest="w_F", type=float, default=1.0)
    g.add_argument("--w-S", dest="w_S", type=float, default=1.0)
    g.add_argument("--target-loss", type=float, default=0.0)
    g.add_argument("--patience",    type=int,   default=0)
    g.add_argument("--maxiter",     type=int,   default=1000)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.dataset:
        args.dataset = [Path("vasprun.xml")]

    if args.stride < 1:
        sys.exit(f"error: --stride must be >= 1, got {args.stride}")
    if args.frame_start < 0:
        sys.exit(f"error: --frame-start must be >= 0, got {args.frame_start}")
    if args.frame_end != -1 and args.frame_end <= args.frame_start:
        sys.exit(
            f"error: --frame-end must be > --frame-start or -1 "
            f"(got start={args.frame_start}, end={args.frame_end})"
        )
    if not (0.0 < args.train_ratio < 1.0):
        sys.exit(f"error: --train-ratio must be in (0, 1), got {args.train_ratio}")

    if not args.no_check:
        missing = [str(p) for p in args.dataset if not p.exists()]
        if missing:
            print(
                "warning: the following dataset paths do not exist yet:\n  "
                + "\n  ".join(missing)
                + "\n(generation proceeds; edit controls.toml or create the files later).",
                file=sys.stderr,
            )

    if args.output.exists() and not args.force:
        sys.exit(
            f"error: {args.output} already exists. Re-run with --force to overwrite."
        )

    toml = build_toml(args)
    args.output.write_text(toml)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
