#!/usr/bin/env python3
"""
Generate VASP inputs for the BVFF cell/stress training set (PbTiO3).

All calculations use the 2x2x2 (40-atom) supercell with the SAME INCAR core,
KPOINTS and POTCARs as the P4mm AIMD behind examples/PbTiO3/pbtio3_222_*K.parquet
(VASP/PbTiO3/GGA/P4mm_w_GGA-GS/MD), so that all energies share one reference
(one ref_group) and can be fitted together:
    PBE (POTCAR default), PAW_PBE Pb_d / Ti_pv / O, ENCUT = 500, PREC = Accurate,
    ISMEAR = 0 (SIGMA = 0.2, VASP default), LREAL = Auto, no IVDW, Gamma-only KPOINTS.
Do NOT change the k-mesh to a denser one: the cubic AIMD (4x4x4 k) already sits
on a different energy reference, which is exactly what this set must avoid.
Only what each set needs differs (IBRION/ISIF/NSW/EDIFF[G]); ISIF >= 2 makes VASP
write the stress tensor for every ionic step, and every step of a relaxation
or MD is a usable training frame.

Sets
  B_ref/<phase>/        full relaxation (ISIF=3): Pm-3m, P4mm, Amm2, R3m
                        → equilibrium cells (a0, c0) and energy differences
                          between phases (depth/anisotropy of the double well)
  A_strain/cub_<s>/     Pm-3m at isotropic strain s (fixed cell, ions fixed by symmetry)
  A_strain/tet_<sa>_<sc>/  P4mm at in-plane strain sa and axial strain sc, ions
                        relaxed at fixed cell (ISIF=2) → E(a,c) surface + stress
  D_rattle/<k>/         P4mm AIMD frames with random strain (|ε| <= 3 %) and atomic
                        rattle (σ = 0.04 Å), single-point (NSW=0, ISIF=2)

POTCAR files are not written (license); concatenate Pb_d, Ti_pv, O in that order
into each directory (or set VASP_PP_PATH and use your usual workflow).

Usage
  python gen_dft_dataset.py --out dft_bvff --aimd examples/PbTiO3/pbtio3_222_300K.parquet
  # after B_ref/P4mm and B_ref/Pm-3m finish, regenerate set A around the relaxed cells:
  python gen_dft_dataset.py --out dft_bvff --only A --a0 <a_P4mm> --c0 <c_P4mm> --acub <a_cubic>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

INCAR_CORE = {
    "PREC": "Accurate", "ENCUT": 500, "ISMEAR": 0, "SIGMA": 0.2, "LREAL": "Auto",
    "EDIFF": 1e-6, "ISYM": 0, "LWAVE": ".FALSE.", "LCHARG": ".FALSE.", "NELM": 120,
}
KPOINTS = "Gamma-only (same as P4mm AIMD)\n0\nGamma\n1 1 1\n0 0 0\n"
ORDER = ["Pb", "Ti", "O"]           # POTCAR order


def unit(a, c, zTi=0.5, zO1=0.0, zO2=0.5, shift=(0.0, 0.0, 0.0)):
    """5-atom PbTiO3 cell; polar displacements along z (or along `shift` for Amm2/R3m)."""
    s = np.array(shift)
    pos = [[0, 0, 0], [.5, .5, zTi], [.5, .5, zO1], [.5, 0, zO2], [0, .5, zO2]]
    pos = np.array(pos, float)
    if np.any(s):                    # generic polar direction: Ti +s, O −s/2 (relaxation refines)
        pos[1] += s; pos[2:] -= 0.5 * s
    return Atoms("PbTiO3", cell=np.diag([a, a, c]), scaled_positions=pos % 1.0, pbc=True)


def sorted_supercell(at: Atoms) -> Atoms:
    sc = at.repeat((2, 2, 2))
    idx = sorted(range(len(sc)), key=lambda i: ORDER.index(sc[i].symbol))
    return sc[idx]


def write_calc(d: Path, at: Atoms, incar: dict, note: str):
    d.mkdir(parents=True, exist_ok=True)
    write(d / "POSCAR", at, format="vasp", direct=True, sort=False, vasp5=True)
    inc = {**INCAR_CORE, **incar, "SYSTEM": note}
    (d / "INCAR").write_text("".join(f"{k} = {v}\n" for k, v in inc.items()))
    (d / "KPOINTS").write_text(KPOINTS)


def set_B(out: Path, a0: float, c0: float, acub: float):
    relax = {"IBRION": 2, "ISIF": 3, "NSW": 200, "EDIFFG": -1e-3}
    phases = {
        "Pm-3m": unit(acub, acub),
        "P4mm":  unit(a0, c0, zTi=0.552, zO1=0.130, zO2=0.661),
        "Amm2":  unit(acub, acub, shift=(0.03, 0.03, 0.0)),
        "R3m":   unit(acub, acub, shift=(0.03, 0.03, 0.03)),
    }
    for name, at in phases.items():
        # ISYM = 0 in the core: symmetry is kept only by the starting geometry;
        # the relaxation trajectory is itself training data.
        write_calc(out / "B_ref" / name, sorted_supercell(at), relax, f"B_ref {name}")


def set_A(out: Path, a0: float, c0: float, acub: float):
    for s in (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06):
        at = unit(acub * (1 + s), acub * (1 + s))
        write_calc(out / "A_strain" / f"cub_{s:+.2f}", sorted_supercell(at),
                   {"IBRION": -1, "ISIF": 2, "NSW": 0}, f"A cubic s={s:+.2f}")
    relax_ions = {"IBRION": 2, "ISIF": 2, "NSW": 100, "EDIFFG": -1e-3}
    for sa in (-0.04, -0.02, 0.0, 0.02, 0.04):
        for sc in (-0.08, -0.04, 0.0, 0.04, 0.08):
            at = unit(a0 * (1 + sa), c0 * (1 + sc), zTi=0.552, zO1=0.130, zO2=0.661)
            write_calc(out / "A_strain" / f"tet_a{sa:+.2f}_c{sc:+.2f}", sorted_supercell(at),
                       relax_ions, f"A P4mm sa={sa:+.2f} sc={sc:+.2f}")


def read_aimd(path: str) -> list[Atoms]:
    """AIMD frames from a BVFF parquet (one row per atom, grouped by Step) or a vasprun.xml."""
    if not path.endswith(".parquet"):
        from ase.io import read
        return read(path, index="100::")             # skip equilibration
    import pandas as pd
    df = pd.read_parquet(path, columns=["Step", "Element", "X", "Y", "Z"] +
                         [f"Lattice_{v}{c}" for v in "abc" for c in "xyz"])
    frames = []
    for _, g in df.groupby("Step", sort=True):
        cell = g.iloc[0][[f"Lattice_{v}{c}" for v in "abc" for c in "xyz"]].to_numpy(float)
        frames.append(Atoms(list(g["Element"]), positions=g[["X", "Y", "Z"]].to_numpy(float),
                            cell=cell.reshape(3, 3), pbc=True))
    return frames                                    # parquet already excludes equilibration


def set_D(out: Path, aimd: str, n: int, seed: int):
    frames = read_aimd(aimd)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(frames), size=n, replace=False)
    for k, i in enumerate(sorted(picks)):
        at = frames[i].copy()
        eps = rng.uniform(-0.03, 0.03, size=(3, 3)); eps = 0.5 * (eps + eps.T)
        at.set_cell(at.cell.array @ (np.eye(3) + eps).T, scale_atoms=True)
        at.positions += rng.normal(scale=0.04, size=at.positions.shape)
        idx = sorted(range(len(at)), key=lambda j: ORDER.index(at[j].symbol))
        write_calc(out / "D_rattle" / f"{k:03d}", at[idx],
                   {"IBRION": -1, "ISIF": 2, "NSW": 0}, f"D rattle frame={i} k={k}")


def main():
    from bvff.core.outputs import init_cli_output
    init_cli_output()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dft_bvff")
    ap.add_argument("--aimd", default="examples/PbTiO3/pbtio3_222_300K.parquet",
                    help="AIMD source for set D: BVFF parquet or vasprun.xml")
    ap.add_argument("--a0", type=float, default=3.8785, help="P4mm a (Å); default = AIMD cell / 2")
    ap.add_argument("--c0", type=float, default=4.1839, help="P4mm c (Å); default = AIMD cell / 2")
    ap.add_argument("--acub", type=float, default=3.97, help="cubic a (Å) starting guess")
    ap.add_argument("--n-rattle", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["A", "B", "D"], nargs="*", default=["A", "B", "D"])
    a = ap.parse_args()
    out = Path(a.out)
    if "B" in a.only: set_B(out, a.a0, a.c0, a.acub)
    if "A" in a.only: set_A(out, a.a0, a.c0, a.acub)
    if "D" in a.only: set_D(out, a.aimd, a.n_rattle, a.seed)
    n = sum(1 for _ in out.rglob("INCAR"))
    print(f"{n} calculation directories under {out}/  (add POTCAR: Pb_d, Ti_pv, O)")


if __name__ == "__main__":
    main()
