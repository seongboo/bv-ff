"""
MD throughput benchmark: Python BVFF (ASE) vs the LAMMPS export.

Run from a fit directory (controls.toml + output/fitted_parameters.toml):

    python3 scripts/benchmark.py --lmp /path/to/lmp --reps 2 3 4 --steps 50

For each supercell size it times an NVE run and reports steps/s and
atom-steps/s. The LAMMPS side runs the exported eam/fs + coul/long + bvv
tables — i.e., the exact same physics — through compiled neighbor lists.
Compare against your NequIP-in-LAMMPS (ML-IAP) logs for the ML baseline.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def bench_python(calc, rep: int, steps: int, a_ref: float, A_site: str) -> dict:
    from ase import units
    from ase.md.verlet import VelocityVerlet
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from scripts.ferroelectric import ideal_perovskite

    atoms = ideal_perovskite(a=a_ref, rep=(rep, rep, rep), A=A_site)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=300,
                                 rng=np.random.RandomState(1))
    dyn = VelocityVerlet(atoms, timestep=2.0 * units.fs)
    dyn.run(2)                      # warm caches outside the timed region
    t0 = time.perf_counter()
    dyn.run(steps)
    dt = time.perf_counter() - t0
    return {"n": len(atoms), "steps_per_s": steps / dt,
            "atom_steps_per_s": steps * len(atoms) / dt}


def bench_lammps(lmp: str, export_dir: Path, rep: int, steps: int,
                 a_ref: float, A_site: str, fitted, elems) -> dict:
    import subprocess
    from types import SimpleNamespace
    from scripts.ferroelectric import ideal_perovskite
    from scripts.export_lammps import write_data

    atoms = ideal_perovskite(a=a_ref, rep=(rep, rep, rep), A=A_site)
    frame = SimpleNamespace(
        lattice=np.asarray(atoms.cell.array),
        positions=atoms.get_scaled_positions(),
        species=atoms.get_chemical_symbols(),
    )
    data = export_dir / f"bench_{rep}.data"
    write_data(frame, elems, fitted.coulomb.charges, data)

    base = (export_dir / "in.bvff").read_text()
    lines = [ln for ln in base.splitlines()
             if not ln.startswith(("dump", "run", "read_data"))]
    lines.insert(lines.index("boundary p p p") + 1, f"read_data {data.name}")
    lines += [
        "velocity all create 300.0 1 dist gaussian",
        "fix 1 all nve",
        "timestep 0.002",
        f"run {steps}",
    ]
    inb = export_dir / f"in.bench_{rep}"
    inb.write_text("\n".join(lines) + "\n")
    log = export_dir / f"log.bench_{rep}"
    res = subprocess.run([lmp, "-in", inb.name, "-log", log.name],
                         cwd=export_dir, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"lmp bench rep={rep} failed:\n{res.stdout[-1500:]}")
    loop = None
    for line in log.read_text().splitlines():
        if line.startswith("Loop time of"):
            loop = float(line.split()[3])
    n = len(atoms)
    return {"n": n, "steps_per_s": steps / loop,
            "atom_steps_per_s": steps * n / loop}


def main() -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="BVFF MD throughput benchmark.")
    ap.add_argument("--lmp", default=None, help="LAMMPS binary (skip if absent)")
    ap.add_argument("--reps", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--export-dir", default="lammps_export")
    args = ap.parse_args()

    from parsers.controls_parser   import parse_controls
    from parsers.parameters_parser import parse_parameters
    from parsers.dataset           import load_dataset
    from src.main                  import build_bvff
    from src.calculator            import BVFFCalculator
    from scripts.ferroelectric     import _detect_chemistry

    controls = parse_controls("controls.toml")
    fitted   = parse_parameters("output/fitted_parameters.toml", validate=True)
    frame    = load_dataset(entries=controls.dataset).frames[0]
    A_site, a_ref = _detect_chemistry(frame)
    elems    = sorted(set(frame.species))
    calc     = BVFFCalculator(build_bvff(controls, fitted))

    print(f"{'system':>10s} {'atoms':>6s} | {'python steps/s':>15s} {'atom-st/s':>10s}"
          f" | {'lammps steps/s':>15s} {'atom-st/s':>10s} | {'speedup':>7s}")
    for rep in args.reps:
        py = bench_python(calc, rep, args.steps, a_ref, A_site)
        row = (f"{rep}x{rep}x{rep}"
               f"{py['n']:>7d} | {py['steps_per_s']:>15.2f} {py['atom_steps_per_s']:>10.0f}")
        if args.lmp:
            lm = bench_lammps(args.lmp, Path(args.export_dir), rep, args.steps,
                              a_ref, A_site, fitted, elems)
            row += (f" | {lm['steps_per_s']:>15.2f} {lm['atom_steps_per_s']:>10.0f}"
                    f" | {lm['steps_per_s'] / py['steps_per_s']:>6.1f}x")
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
