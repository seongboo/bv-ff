"""
LAMMPS export writers: the eam/fs tables must reproduce the Python term
functions numerically (they are generated THROUGH those functions, so this
guards the file layout and unit conventions, not re-derives the physics).
Running LAMMPS itself is exercised by bvff/tools/export_lammps.py --validate;
these tests need no LAMMPS binary.
"""
from __future__ import annotations

import numpy as np
import pytest

from bvff.parsers.parameters_parser import (
    Parameters, CoulombParams, BuckinghamParams, BuckinghamPair,
    BVParams, BVSpecies, BVPair, BVVParams, BVVSpecies,
)
from bvff.tools.export_lammps import write_eam_fs, write_bvv, NRHO, NR, RHO_MAX
from bvff.core.potentials import BV, Buckingham


def _params() -> Parameters:
    p = Parameters(cutoff=6.0, smooth_width=1.0)
    p.coulomb    = CoulombParams(charges={"Pb": 1.4, "Ti": 1.0, "O": -0.8})
    p.buckingham = BuckinghamParams(pairs={
        "O-O":  BuckinghamPair(A=24850.0, rho=0.223, C=32.5),
        "O-Pb": BuckinghamPair(A=5732.0, rho=0.255, C=1e-5),
    })
    p.BV = BVParams(
        species={"Pb": BVSpecies(1.63, 0.146), "Ti": BVSpecies(2.86, 0.172),
                 "O": BVSpecies(3.37, 0.541)},
        pairs={"O-Pb": BVPair(1.965, 6.0, 0.393), "O-Ti": BVPair(1.913, 5.2, 0.390)},
    )
    p.BVV = BVVParams(
        species={"Pb": BVVSpecies(1.55, 0.179), "Ti": BVVSpecies(0.278, 0.098)},
        pairs={"O-Pb": BVPair(2.07, 6.0, 0.496), "O-Ti": BVPair(1.76, 5.2, 0.378)},
    )
    return p


def test_eam_fs_tables_match_python(tmp_path):
    p = _params()
    path = tmp_path / "t.eam.fs"
    write_eam_fs(p, ["O", "Pb", "Ti"], path, bv_form="exp")
    lines = path.read_text().splitlines()

    assert lines[3].split() == ["3", "O", "Pb", "Ti"]
    nrho, drho, nr, dr, cutoff = lines[4].split()
    assert int(nrho) == NRHO and int(nr) == NR and float(cutoff) == 6.0

    # Parse arrays back in file order.
    def block(start, n):
        out = []
        k = start
        while len(out) < n:
            out.extend(float(v) for v in lines[k].split())
            k += 1
        return np.array(out), k

    r      = np.arange(NR) * float(dr)
    rho_ax = np.arange(NRHO) * float(drho)
    idx = 5
    F, rho_tab = {}, {}
    for e in ("O", "Pb", "Ti"):
        idx += 1                        # element header line
        F[e], idx = block(idx, NRHO)
        for e2 in ("O", "Pb", "Ti"):
            rho_tab[(e, e2)], idx = block(idx, NR)
    # phi tables: (O,O), (Pb,O), (Pb,Pb), (Ti,O), (Ti,Pb), (Ti,Ti)
    rphi = {}
    for i, e1 in enumerate(["O", "Pb", "Ti"]):
        for e2 in ["O", "Pb", "Ti"][: i + 1]:
            rphi[(e1, e2)], idx = block(idx, NR)

    # Embedding = S (rho - V0)^2 exactly.
    for e in ("O", "Pb", "Ti"):
        sp = p.BV.species[e]
        np.testing.assert_allclose(F[e], sp.S * (rho_ax - sp.V0) ** 2, rtol=1e-12)

    # Densities reproduce the (tapered, masked) V_ij and vanish at the cutoff.
    bv = BV({a: {"V0": s.V0, "S": s.S} for a, s in p.BV.species.items()},
            {k: {"r0": q.r0, "C": q.C, "b": q.b} for k, q in p.BV.pairs.items()},
            cutoff=6.0, form="exp", smooth_width=1.0)
    rg = np.maximum(r, 1e-6)
    q  = p.BV.pairs["O-Ti"]
    V, _ = bv._valence_and_deriv(rg, np.full_like(rg, q.r0),
                                 np.full_like(rg, q.C), np.full_like(rg, q.b))
    np.testing.assert_allclose(rho_tab[("O", "Ti")], V, rtol=1e-12)
    np.testing.assert_allclose(rho_tab[("Ti", "O")], V, rtol=1e-12)   # symmetric
    assert np.all(rho_tab[("O", "O")] == 0.0)                          # no O-O BV pair
    assert np.all(rho_tab[("Pb", "Ti")] == 0.0)                        # phantom guard
    assert rho_tab[("O", "Ti")][-1] == 0.0                             # taper at r_c

    # Pair table is r*phi with the guarded+tapered Buckingham.
    buck = Buckingham({k: {"A": v.A, "rho": v.rho, "C": v.C}
                       for k, v in p.buckingham.pairs.items()},
                      cutoff=6.0, smooth_width=1.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        phi, _ = buck._pair_phi(["O", "Pb"], np.zeros(NR, dtype=np.int64),
                                np.ones(NR, dtype=np.int64), rg)
    np.testing.assert_allclose(rphi[("Pb", "O")], r * phi, rtol=1e-10, atol=1e-12)
    assert np.all(rphi[("Ti", "Ti")] == 0.0)
    assert np.isfinite(rphi[("O", "O")]).all()                         # guard at r→0


def test_bvv_file_roundtrip(tmp_path):
    p = _params()
    path = tmp_path / "t.bvv"
    write_bvv(p, path, bv_form="exp")
    txt = path.read_text()
    assert "cutoff        6.0" in txt
    assert "form          exp" in txt
    assert "species       2" in txt and "pairs         2" in txt
    # Parse like the C++ reader: name W0 D / name1 name2 r0 C b
    got = {}
    for ln in txt.splitlines():
        tok = ln.split("#")[0].split()
        if len(tok) == 3 and tok[0] in ("Pb", "Ti"):
            got[tok[0]] = (float(tok[1]), float(tok[2]))
    assert got["Pb"] == (1.55, 0.179) and got["Ti"] == (0.278, 0.098)
