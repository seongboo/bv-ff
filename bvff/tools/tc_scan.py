"""
Ferroelectric transition-temperature (Tc) scan for a fitted BVFF.

Runs MD at a ladder of temperatures and tracks the ferroelectric order
parameter — the coherent cation off-centering ⟨u⟩ (origin-free, robust in MD,
see ``ferroelectric.cation_offcentering``) — plus, under ``--npt``, the cell
tetragonality c/a. Ferroelectric order melts where |⟨u_z⟩| collapses and
c/a → 1; that crossover is the model's Tc (experiment: PbTiO3 763 K,
BaTiO3 ~400 K).

NPT uses the Parrinello-Rahman-style ``ase.md.npt.NPT`` barostat, which needs
the virial stress every step — cheap since every BVFF term has an analytic
virial (Phase 1). NVT (default) holds the training-frame cell fixed: faster
and stable, but it suppresses the strain coupling, so the NVT "Tc" is only a
qualitative bracket; use --npt for the physical estimate.

Run from a fit directory (controls.toml + output/fitted_parameters.toml):

    bvff-tc-scan --temps 100 300 500 700 900 --steps 4000 --npt

Writes tc_scan.csv and tc_scan.png into the run's output directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ferroelectric import cation_offcentering


def scan_temperatures(
    atoms0,
    calc,
    temps,
    steps:        int   = 3000,
    equil:        int   = 1500,
    timestep_fs:  float = 2.0,
    sample_every: int   = 20,
    npt:          bool  = False,
    bulk_modulus_GPa: float = 100.0,
    seed:         int   = 0,
    logger              = None,
) -> list[dict]:
    """
    MD at each temperature; after ``equil`` steps, sample the Ti off-centering
    vector (and the cell under NPT) every ``sample_every`` steps.

    Returns one record per temperature:
      {"T", "u_mean" (3,), "u_abs_mean", "u_abs_std", "c_over_a" (NPT only),
       "n_samples", "exploded"}

    Each temperature restarts from ``atoms0`` (a polar, ideally pre-relaxed
    configuration) rather than chaining — chaining biases the scan by hysteresis;
    independent starts probe where the polar state is *thermodynamically* lost.
    """
    import time
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from bvff.core.outputs import LogThrottle

    log = logger.info if logger is not None else (lambda msg: print(msg, flush=True))
    records = []
    for k, T in enumerate(temps):
        atoms = atoms0.copy()
        atoms.calc = calc
        rng = np.random.default_rng(seed + k)
        MaxwellBoltzmannDistribution(atoms, temperature_K=float(T), rng=rng)

        if npt:
            from ase.md.npt import NPT
            # ttime ~ 25 fs thermostat; pfactor = ptime² · B with a generic
            # perovskite bulk modulus — only the relaxation timescale, not the
            # physics, depends on B here.
            dyn = NPT(atoms, timestep_fs * units.fs, temperature_K=float(T),
                      externalstress=0.0, ttime=25.0 * units.fs,
                      pfactor=(75.0 * units.fs) ** 2 * bulk_modulus_GPa * units.GPa)
        else:
            from ase.md.langevin import Langevin
            dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=float(T),
                           friction=0.01, rng=rng)

        u_samples, ca_samples = [], []

        def _sample():
            if dyn.nsteps >= equil:
                u_samples.append(cation_offcentering(atoms, "Ti"))
                if npt:
                    cell = atoms.cell.lengths()
                    ca_samples.append(cell[2] / np.mean(cell[:2]))

        throttle, t0 = LogThrottle(), time.time()

        def _progress():
            n = dyn.nsteps
            if n and throttle.ready(force=(n == steps)):
                el = time.time() - t0
                log(f"    T={T:6.0f} K | step {n:6d}/{steps} | "
                    f"T_inst {atoms.get_temperature():7.1f} K | elapsed {el:6.1f}s "
                    f"| ETA {el / n * (steps - n):6.1f}s")

        dyn.attach(_sample, interval=sample_every)
        dyn.attach(_progress, interval=1)
        dyn.run(steps)

        u  = np.asarray(u_samples) if u_samples else np.zeros((0, 3))
        ok = bool(np.isfinite(atoms.get_positions()).all() and
                  (u.size == 0 or np.isfinite(u).all()))
        rec = {
            "T":          float(T),
            "u_mean":     u.mean(axis=0).tolist() if len(u) else [0.0, 0.0, 0.0],
            "u_abs_mean": float(np.linalg.norm(u, axis=1).mean()) if len(u) else 0.0,
            "u_abs_std":  float(np.linalg.norm(u, axis=1).std()) if len(u) else 0.0,
            "n_samples":  int(len(u)),
            "exploded":   not ok,
        }
        if npt:
            rec["c_over_a"] = float(np.mean(ca_samples)) if ca_samples else float("nan")
        records.append(rec)
        log(f"  T={T:6.0f} K | <|u|> = {rec['u_abs_mean']:.3f} ± "
            f"{rec['u_abs_std']:.3f} Å | <u_z> = {rec['u_mean'][2]:+.3f} Å"
            + (f" | c/a = {rec.get('c_over_a', float('nan')):.4f}" if npt else "")
            + (" | EXPLODED" if rec["exploded"] else ""))
    return records


def estimate_tc(records) -> float | None:
    """Crude Tc bracket: first temperature where the coherent polar component
    |⟨u_z⟩| drops below half its lowest-temperature value (None if it never
    does — Tc above the scanned range)."""
    if not records:
        return None
    u0 = abs(records[0]["u_mean"][2])
    if u0 < 0.05:
        return None
    for rec in records[1:]:
        if abs(rec["u_mean"][2]) < 0.5 * u0:
            return rec["T"]
    return None


def _main() -> int:
    import argparse
    import csv
    import logging

    from bvff.core.outputs import init_cli_output
    init_cli_output()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("bvff.tc")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[100, 300, 500, 700, 900])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--equil", type=int, default=1500)
    ap.add_argument("--timestep-fs", type=float, default=2.0)
    ap.add_argument("--npt", action="store_true",
                    help="NPT (Parrinello-Rahman) instead of fixed-cell NVT")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from bvff.parsers.controls_parser   import parse_controls
    from bvff.parsers.parameters_parser import parse_parameters
    from bvff.parsers.dataset           import load_dataset
    from bvff.core.main                  import build_bvff
    from bvff.core.calculator            import BVFFCalculator, atoms_from_frame

    controls = parse_controls("controls.toml")
    fitted   = parse_parameters("output/fitted_parameters.toml", validate=True)
    calc     = BVFFCalculator(build_bvff(controls, fitted))

    # Start from a polar training frame (already in the ferroelectric basin).
    frame  = load_dataset(entries=controls.dataset, logger=log).frames[0]
    atoms0 = atoms_from_frame(frame)
    log.info(f"Start: {len(atoms0)} atoms | "
             f"|u0| = {np.linalg.norm(cation_offcentering(atoms0, 'Ti')):.3f} Å "
             f"| {'NPT' if args.npt else 'NVT'} | temps = {args.temps}")

    records = scan_temperatures(
        atoms0, calc, args.temps, steps=args.steps, equil=args.equil,
        timestep_fs=args.timestep_fs, npt=args.npt, seed=args.seed, logger=log,
    )

    out = Path(controls.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = sorted({k for r in records for k in r})
    with open(out / "tc_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)

    tc = estimate_tc(records)
    log.info(f"Tc bracket (|<u_z>| half-collapse): "
             + (f"~{tc:.0f} K" if tc else "above scanned range"))

    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        T  = [r["T"] for r in records]
        uz = [abs(r["u_mean"][2]) for r in records]
        ua = [r["u_abs_mean"] for r in records]
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(T, uz, "o-", label="|⟨u_z⟩| (coherent)")
        ax.plot(T, ua, "s--", label="⟨|u|⟩ (local)")
        if tc:
            ax.axvline(tc, color="k", ls=":", lw=1, label=f"Tc ~ {tc:.0f} K")
        ax.set_xlabel("T (K)")
        ax.set_ylabel("Ti off-centering (Å)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "tc_scan.png", dpi=150)
    except Exception as exc:                                   # pragma: no cover
        log.info(f"  (tc_scan.png skipped: {exc})")

    log.info(f"Wrote {out / 'tc_scan.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
