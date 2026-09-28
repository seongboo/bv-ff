"""
Convert VASP AIMD trajectories (vasprun.xml) into the long-format parquet that
the BVFF dataset loader (bvff/parsers/parquet_reader.py) expects.

One parquet per trajectory; several vasprun.xml files (e.g. step_0-2000,
step_2000-4000, ...) are concatenated in order with continuous Step numbering.
Columns match bvff/parsers/parquet_reader.py / bvff/parsers/vasp_reader.py conventions:

    Step, Atom_Index, Element,
    X, Y, Z                    Cartesian positions (Å)
    Force_X, Force_Y, Force_Z  forces (eV/Å)
    Total_Energy               e_fr_energy of the step (eV), repeated per atom
    Lattice_ax .. Lattice_cz   the 3x3 cell, row-major, repeated per atom
    Stress_xx .. Stress_xy     Voigt virial stress in kBar (VASP convention),
                               written ONLY when every step has stress (so the
                               loader never sees NaN). The BVFF loss converts
                               kBar→eV/Å³ via /1602.18 with VASP's sign flip.

Energy/stress/position conventions are taken straight from pymatgen (same as
bvff/parsers/vasp_reader.py) so vasprun- and parquet-loaded data are identical.

CLI:
    bvff-vasprun2data OUT.parquet vr1/vasprun.xml [vr2/vasprun.xml ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_LATTICE_COLS = [
    "Lattice_ax", "Lattice_ay", "Lattice_az",
    "Lattice_bx", "Lattice_by", "Lattice_bz",
    "Lattice_cx", "Lattice_cy", "Lattice_cz",
]
_STRESS_COLS = [
    "Stress_xx", "Stress_yy", "Stress_zz",
    "Stress_yz", "Stress_xz", "Stress_xy",
]


def _voigt(stress_3x3: np.ndarray) -> list[float]:
    s = np.asarray(stress_3x3, dtype=float)
    return [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]


def convert(vasprun_paths, out_path, log=print) -> dict:
    """Concatenate ``vasprun_paths`` (in order) into one parquet at ``out_path``.

    Returns a summary dict: {steps, atoms, has_stress, rows, out}.
    """
    import pandas as pd
    from pymatgen.io.vasp.outputs import Vasprun

    cols: dict[str, list] = {k: [] for k in (
        ["Step", "Atom_Index", "Element", "X", "Y", "Z",
         "Force_X", "Force_Y", "Force_Z", "Total_Energy"] + _LATTICE_COLS + _STRESS_COLS
    )}
    step = 0
    all_stress = True
    n_atoms = None

    for vp in vasprun_paths:
        vp = str(vp)
        log(f"  parsing {vp} ...")
        vr = Vasprun(vp, parse_dos=False, parse_eigen=False,
                     parse_projected_eigen=False, exception_on_bad_xml=False)
        for st in vr.ionic_steps:
            struct = st["structure"]
            n      = len(struct)
            n_atoms = n_atoms or n
            cart   = np.asarray(struct.cart_coords, dtype=float)
            forces = np.asarray(st["forces"], dtype=float)
            lat    = np.asarray(struct.lattice.matrix, dtype=float).reshape(-1)
            energy = float(st["e_fr_energy"])
            stress = st.get("stress", None)

            cols["Step"].append(np.full(n, step, dtype=np.int64))
            cols["Atom_Index"].append(np.arange(n, dtype=np.int64))
            cols["Element"].append(np.array([str(s) for s in struct.species]))
            cols["X"].append(cart[:, 0]); cols["Y"].append(cart[:, 1]); cols["Z"].append(cart[:, 2])
            cols["Force_X"].append(forces[:, 0]); cols["Force_Y"].append(forces[:, 1]); cols["Force_Z"].append(forces[:, 2])
            cols["Total_Energy"].append(np.full(n, energy))
            for c, v in zip(_LATTICE_COLS, lat):
                cols[c].append(np.full(n, v))
            if stress is not None:
                for c, v in zip(_STRESS_COLS, _voigt(np.asarray(stress))):
                    cols[c].append(np.full(n, v))
            else:
                all_stress = False
            step += 1

    if step == 0:
        raise ValueError(f"no ionic steps found in {list(vasprun_paths)}")

    keep = ["Step", "Atom_Index", "Element", "X", "Y", "Z",
            "Force_X", "Force_Y", "Force_Z", "Total_Energy"] + _LATTICE_COLS
    if all_stress:
        keep += _STRESS_COLS
    df = pd.DataFrame({k: np.concatenate(cols[k]) for k in keep})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy")
    summary = {"steps": step, "atoms": n_atoms, "has_stress": all_stress,
               "rows": len(df), "out": str(out_path)}
    log(f"  wrote {out_path}: {step} steps, {n_atoms} atoms, "
        f"stress={'yes' if all_stress else 'no'}, {len(df)} rows")
    return summary


def main():
    from bvff.core.outputs import init_cli_output
    init_cli_output()
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: bvff-vasprun2data OUT.parquet vasprun1.xml [vasprun2.xml ...]")
        sys.exit(1)
    out = sys.argv[1]
    vrs = sys.argv[2:]
    convert(vrs, out)


if __name__ == "__main__":
    main()
