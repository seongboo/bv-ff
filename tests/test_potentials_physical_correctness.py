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
    ew = Ewald({"Na": 1.0, "Cl": -1.0}, cutoff=2.8, accuracy=1e-9)
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


# ──────────────────────────────────────────────
# 4. Ewald: α independence, cell-size consistency, neutrality
# ──────────────────────────────────────────────

def test_ewald_alpha_independence(frame, params):
    """For a neutral cell the converged Ewald energy is independent of the
    splitting α. Vary α with r_c = p/α and k_c = 2αp at fixed δ."""
    L, sp, x = frame.lattice, frame.species, frame.positions
    delta = 1e-8
    p = np.sqrt(-np.log(delta))
    energies = []
    for alpha in (0.45, 0.60, 0.75):
        ew = Ewald(params.coulomb.charges, cutoff=p / alpha, accuracy=delta)
        assert ew.alpha == pytest.approx(alpha)
        energies.append(ew.energy(L, sp, x) / len(sp))
    assert np.ptp(energies) < 1e-6, f"E/atom spread over α: {np.ptp(energies):.2e} eV/atom"


def test_ewald_same_accuracy_for_supercell(frame, params):
    """k_c is a |k| sphere, so the same (r_c, δ) must give the same E/atom for
    the cell and its 2×2×2 supercell (an index-based kmax would not)."""
    L, sp, x = frame.lattice, frame.species, frame.positions
    ew = Ewald(params.coulomb.charges, cutoff=6.0, accuracy=1e-8)
    L2, sp2, x2 = _supercell(L, sp, x)
    e1 = ew.energy(L,  sp,  x)  / len(sp)
    e2 = ew.energy(L2, sp2, x2) / len(sp2)
    assert e2 == pytest.approx(e1, abs=1e-6)


def test_ewald_rejects_non_neutral_cell(frame):
    ew = Ewald({"Pb": 1.4, "Ti": 1.0, "O": -0.7}, cutoff=6.0)   # net +2.4 e per cell
    with pytest.raises(ValueError, match="charge-neutral"):
        ew.energy(frame.lattice, frame.species, frame.positions)


def test_ewald_finite_epsilon_warns():
    with pytest.warns(UserWarning, match="epsilon"):
        Ewald({"Na": 1.0, "Cl": -1.0}, cutoff=6.0, epsilon=1.0)


# ──────────────────────────────────────────────
# 5. Fitting keeps the charges neutral
# ──────────────────────────────────────────────

def test_fitting_charge_neutrality(frame, params):
    from parsers.controls_parser import parse_controls
    from src.fitting import (
        params_to_vector, vector_to_params, frames_composition, dependent_species,
    )
    pc   = parse_controls(str(_ROOT / "controls.toml")).potentials
    comp = frames_composition([frame])
    assert dependent_species(comp) == "O"

    x, keys = params_to_vector(params, pc, comp)
    assert "coulomb.O" not in keys and "coulomb.Pb" in keys

    rng = np.random.default_rng(1)
    for _ in range(5):
        x_trial = x.copy()
        for k, name in enumerate(keys):
            if name.startswith("coulomb."):
                x_trial[k] = rng.uniform(-3, 3)
        q = vector_to_params(x_trial, keys, params, comp).coulomb.charges
        assert sum(n * q[s] for s, n in comp.items()) == pytest.approx(0.0, abs=1e-12)


def test_frames_composition_rejects_mixed_ratios(frame):
    from dataclasses import replace
    from src.fitting import frames_composition
    other = replace(frame, species=["Pb"] * 8 + ["Ti"] * 8 + ["O"] * 23 + ["Pb"])
    with pytest.raises(ValueError, match="composition"):
        frames_composition([frame, other])


# ──────────────────────────────────────────────
# 6. Functional forms against closed-form cases
# ──────────────────────────────────────────────

def _dimer(d, box=20.0):
    """Ti at the origin, O at distance d along x, in a cubic box large enough
    that no periodic image is inside the cutoff."""
    L = np.eye(3) * box
    x = np.array([[0.0, 0.0, 0.0], [d / box, 0.0, 0.0]])
    return L, ["Ti", "O"], x


def test_repulsive_dimer_closed_form():
    """E_r = (B/d)^12 and |F| = 12 B^12 / d^13, repulsive (pushes atoms apart)."""
    from src.potentials import Repulsive
    B, d = 1.28, 1.9
    L, sp, x = _dimer(d)
    rep = Repulsive({"O-Ti": B}, cutoff=6.0)
    assert rep.energy(L, sp, x) == pytest.approx((B / d) ** 12, rel=1e-12)
    f = rep.forces(L, sp, x)
    fmag = 12.0 * B**12 / d**13
    np.testing.assert_allclose(f[1], [ fmag, 0, 0], rtol=1e-12, atol=1e-15)   # O pushed to +x
    np.testing.assert_allclose(f[0], [-fmag, 0, 0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("form", ["power", "exp"])
def test_bv_only_parameterized_pairs(frame, params, form):
    """Bond valence sums include only pairs with BV parameters (cation–O).
    Compare get_valence with an explicit sum restricted to those pairs."""
    from src.potentials import BV
    L, sp, x = frame.lattice, frame.species, frame.positions
    pp = {k: {"r0": v.r0, "C": v.C, "b": v.b} for k, v in params.BV.pairs.items()}
    ss = {a: {"V0": s.V0, "S": s.S} for a, s in params.BV.species.items()}
    bv = BV(ss, pp, cutoff=6.0, form=form)
    V  = bv.get_valence(L, sp, x)

    i, j, rv = bv._neighbors(L, x, 6.0)
    r = np.linalg.norm(rv, axis=1)
    V_ref = np.zeros(len(sp))
    for a, b_, rr in zip(i, j, r):
        p = pp.get(f"{sp[a]}-{sp[b_]}") or pp.get(f"{sp[b_]}-{sp[a]}")
        if p is None:
            continue
        V_ref[a] += (p["r0"] / rr) ** p["C"] if form == "power" else np.exp((p["r0"] - rr) / p["b"])
    np.testing.assert_allclose(V, V_ref, rtol=1e-12, atol=1e-14)


def test_bvv_exp_ignores_unparameterized_pairs():
    """A lone cation–cation pair (no BV parameters) must give W = 0 and zero
    force in the exp form."""
    from src.potentials import BVV
    L = np.eye(3) * 20.0
    x = np.array([[0.0, 0.0, 0.0], [3.9 / 20.0, 0.0, 0.0]])
    bvv = BVV({"Pb": {"W0": 0.5, "D": 0.1}, "Ti": {"W0": 0.3, "D": 0.1}},
              {"O-Ti": {"r0": 1.8, "C": 5.2, "b": 0.37}}, cutoff=6.0, form="exp")
    W = bvv.get_bvv(L, ["Pb", "Ti"], x)
    assert np.all(W == 0.0)
    assert np.all(bvv.forces(L, ["Pb", "Ti"], x) == 0.0)
