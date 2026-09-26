"""
Physical-correctness checks: is the energy *itself* right, not merely
self-consistent with its own forces?

The finite-difference / invariance / stress tests verify that forces and
stress are exact derivatives of whatever energy the code computes. They
cannot detect an energy expression that is consistently wrong. The checks
here compare against conditions every correct periodic potential satisfies:

1. Supercell invariance — E is extensive: E(n×n×n supercell) = n³·E(cell),
   and replicated atoms feel the same force. Fails if the neighbor list
   misses periodic images (cutoff > half the perpendicular cell width) or
   if a term depends on how the cell is chosen.
2. Ewald accuracy — the production Ewald term (settings from build_bvff)
   matches a converged, independent reference (pymatgen EwaldSummation,
   tin-foil boundary). A positive control on NaCl (Madelung constant) with
   converged settings isolates class bugs from settings bugs.
3. Wrap invariance — moving an atom by a lattice vector is the same
   physical configuration, so E and F must not change. Fails for any term
   that depends on unwrapped Cartesian positions (e.g. a dipole surface
   term computed from wrapped coordinates).

Tests use the production configuration (controls.toml + parameters.toml via
build_bvff), so they check what an actual fit / MD run would use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from parsers.dataset           import load_dataset, DatasetEntry
from parsers.controls_parser   import parse_controls
from parsers.parameters_parser import parse_parameters
from src.main                  import build_bvff
from src.potentials            import clear_neighbor_cache
from src.extensions.ewald      import Ewald, clear_kvec_cache


_ROOT = Path(__file__).resolve().parent.parent

# Per-term tolerances (eV/atom for energy, eV/Å for forces).
E_TOL = {"Ewald": 1e-4, "Repulsive": 1e-6, "BV": 1e-6, "BVV": 1e-6}
F_TOL = {"Ewald": 1e-3, "Repulsive": 1e-6, "BV": 1e-5, "BVV": 1e-5}
TERMS = list(E_TOL)


# ──────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────

@pytest.fixture(scope="module")
def frame():
    clear_neighbor_cache()
    clear_kvec_cache()
    data = load_dataset(entries=[DatasetEntry(
        path        = str(_ROOT / "examples/PbTiO3/vasprun.xml"),
        frame_start = 100, frame_end = 101, stride = 1,
    )])
    return data.frames[0]


@pytest.fixture(scope="module")
def params():
    return parse_parameters(str(_ROOT / "parameters.toml"))


@pytest.fixture(scope="module")
def bvff(params):
    controls = parse_controls(str(_ROOT / "controls.toml"))
    return build_bvff(controls, params)


def _term(bvff, name):
    for t in bvff.terms:
        if type(t).__name__ == name:
            return t
    pytest.skip(f"{name} not enabled in controls.toml")


def _supercell(lattice, species, positions, m=(2, 2, 2)):
    """Replicate the cell m[0]×m[1]×m[2]. Copy k of atom a has index
    k*N + a, so forces of all copies can be compared to the original."""
    m      = np.asarray(m)
    shifts = np.array([[a, b, c] for a in range(m[0])
                                 for b in range(m[1])
                                 for c in range(m[2])])
    pos = np.vstack([(positions + s) / m for s in shifts])
    return lattice * m[:, None], list(species) * len(shifts), pos


# ──────────────────────────────────────────────
# 1. Supercell invariance
# ──────────────────────────────────────────────

@pytest.mark.parametrize("name", TERMS)
def test_supercell_energy_per_atom(bvff, frame, name):
    t = _term(bvff, name)
    L, sp, x = frame.lattice, frame.species, frame.positions
    L2, sp2, x2 = _supercell(L, sp, x)
    e1 = t.energy(L,  sp,  x)  / len(sp)
    e2 = t.energy(L2, sp2, x2) / len(sp2)
    assert e2 == pytest.approx(e1, abs=E_TOL[name]), (
        f"{name}: E/atom differs between cell ({e1:+.6e}) and 2x2x2 "
        f"supercell ({e2:+.6e}); diff {e2 - e1:+.2e} eV/atom"
    )


@pytest.mark.parametrize("name", TERMS)
def test_supercell_forces(bvff, frame, name):
    t = _term(bvff, name)
    L, sp, x = frame.lattice, frame.species, frame.positions
    L2, sp2, x2 = _supercell(L, sp, x)
    f1 = t.forces(L,  sp,  x)
    f2 = t.forces(L2, sp2, x2).reshape(-1, len(sp), 3)   # (copies, N, 3)
    err = np.max(np.abs(f2 - f1[None]))
    assert err < F_TOL[name], (
        f"{name}: max |F(supercell copy) - F(cell)| = {err:.2e} eV/Å"
    )


# ──────────────────────────────────────────────
# 2. Ewald accuracy
# ──────────────────────────────────────────────

def test_ewald_class_madelung_nacl():
    """Positive control: with converged settings the Ewald class reproduces
    the NaCl Madelung constant. Uses a cutoff below half the cell width so
    the minimum-image neighbor list is valid. If this passes while the
    production test below fails, the problem is the settings, not the math."""
    a  = 5.64
    L  = np.eye(3) * a
    na = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    cl = [[.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]
    x  = np.array(na + cl, dtype=float)
    sp = ["Na"] * 4 + ["Cl"] * 4
    ew = Ewald({"Na": 1.0, "Cl": -1.0}, alpha=1.5, kmax=12,
               cutoff=2.8, epsilon=np.inf)
    e_pair  = ew.energy(L, sp, x) / 4                   # per NaCl formula unit
    madelung = -e_pair * (a / 2) / Ewald.KE
    assert madelung == pytest.approx(1.747565, abs=1e-5)


def test_ewald_production_vs_pymatgen(bvff, frame, params):
    """Production Ewald term (as built by build_bvff) vs a converged
    reference. pymatgen's EwaldSummation uses the tin-foil boundary."""
    from pymatgen.core import Structure
    from pymatgen.analysis.ewald import EwaldSummation

    t = _term(bvff, "Ewald")
    L, sp, x = frame.lattice, frame.species, frame.positions
    s = Structure(L, sp, x)
    s.add_oxidation_state_by_element(params.coulomb.charges)
    ref = EwaldSummation(s, acc_factor=12.0, compute_forces=True)

    n = len(sp)
    e_code, f_code = t.energy_and_forces(L, sp, x)
    assert e_code / n == pytest.approx(ref.total_energy / n, abs=E_TOL["Ewald"]), (
        f"Ewald E/atom {e_code / n:+.6f} vs reference {ref.total_energy / n:+.6f}"
    )
    err = np.max(np.abs(f_code - ref.forces))
    assert err < F_TOL["Ewald"], f"Ewald max force error {err:.2e} eV/Å"


# ──────────────────────────────────────────────
# 3. Wrap invariance
# ──────────────────────────────────────────────

_LATTICE_SHIFTS = [
    (0, np.array([1, 0, 0])),
    (8, np.array([0, -1, 0])),
    (16, np.array([1, 1, -1])),
]


@pytest.mark.parametrize("name", TERMS)
@pytest.mark.parametrize("atom,shift", _LATTICE_SHIFTS)
def test_wrap_invariance(bvff, frame, name, atom, shift):
    t = _term(bvff, name)
    L, sp, x = frame.lattice, frame.species, frame.positions
    x_shift        = x.copy()
    x_shift[atom] += shift                      # same physical configuration
    e0, f0 = t.energy(L, sp, x),       t.forces(L, sp, x)
    e1, f1 = t.energy(L, sp, x_shift), t.forces(L, sp, x_shift)
    n = len(sp)
    assert e1 / n == pytest.approx(e0 / n, abs=1e-8), (
        f"{name}: E/atom changed by {(e1 - e0) / n:+.3e} eV/atom when atom "
        f"{atom} was moved by lattice vector {shift}"
    )
    err = np.max(np.abs(f1 - f0))
    assert err < 1e-8, f"{name}: forces changed by up to {err:.2e} eV/Å"
