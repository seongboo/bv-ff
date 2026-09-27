"""
BVV must work independently of BV.

Before the fix, BVV borrowed its bond-valence pair parameters (r0, C/b) from
``params.BV.pairs`` in ``build_bvff`` and ``params_to_vector`` only emitted
BVV's per-species W0/D — so with ``use_BV=0`` the BVV pair params were either
frozen (file carried [BV.pairs]) or empty (auto-generated → V_ij=0 → zero
forces, degenerate). BVV now owns its pairs end to end: parsed, generated,
fit, bounded, and built independently, while still inheriting a *copy* of
[BV.pairs] when [BVV.pairs] is omitted (backward compatibility).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from bvff.parsers.parameters_parser import (
    parse_parameters, generate_parameters, save_parameters,
    Parameters, BVVParams, BVParams, BVPair, BVVSpecies,
)
from bvff.core.fitting import params_to_vector, build_bounds


SPECIES = ["Pb", "Ti", "O"]


class _PC:
    """Minimal stand-in for controls.potentials with only BVV enabled."""
    use_coulomb = use_repulsive = use_buckingham = use_BV = use_angle = False
    use_BVV = True
    bv_form = "power"


# ──────────────────────────────────────────────
# Auto-generation: BVV gets its own pairs
# ──────────────────────────────────────────────

def test_generate_bvv_alone_has_pairs():
    p = generate_parameters(SPECIES, use_coulomb=False, use_repulsive=False,
                            use_BV=False, use_BVV=True, use_angle=False)
    assert p.BVV.species, "BVV species missing"
    assert p.BVV.pairs, "BVV pairs must be auto-generated even with use_BV=0"
    # cation–anion pairs only (no Pb-Ti / Pb-Pb / Ti-Ti)
    assert all("O" in pair for pair in p.BVV.pairs)


# ──────────────────────────────────────────────
# Fit vector / bounds include BVV pairs and stay aligned
# ──────────────────────────────────────────────

def test_bvv_pairs_in_fit_vector_and_bounds_aligned():
    p = generate_parameters(SPECIES, use_coulomb=False, use_repulsive=False,
                            use_BV=False, use_BVV=True, use_angle=False)
    vec, keys = params_to_vector(p, _PC)
    assert any(k.startswith("BVV.pairs.") and k.endswith(".r0") for k in keys)
    assert any(k.startswith("BVV.pairs.") and k.endswith(".C") for k in keys)
    # dual_annealing requires one bound per parameter.
    assert len(build_bounds(keys)) == len(vec) == len(keys)


def test_bvv_pairs_fit_b_when_exp_form():
    class _PCExp(_PC):
        bv_form = "exp"
    p = generate_parameters(SPECIES, use_coulomb=False, use_repulsive=False,
                            use_BV=False, use_BVV=True, use_angle=False)
    _, keys = params_to_vector(p, _PCExp)
    assert any(k.endswith(".b") for k in keys if k.startswith("BVV.pairs."))
    assert not any(k.endswith(".C") for k in keys if k.startswith("BVV.pairs."))


# ──────────────────────────────────────────────
# Backward compatibility: inherit a *copy* of BV pairs
# ──────────────────────────────────────────────

def test_bvv_inherits_copy_of_bv_pairs(tmp_path):
    f = tmp_path / "old.toml"
    f.write_text(
        "cutoff = 6.0\n"
        "[BV.pairs.O-Pb]\nr0 = 2.06\nC = 6.0\n"
        "[BV.pairs.O-Ti]\nr0 = 1.81\nC = 5.2\n"
        "[BVV.Pb]\nW0 = 0.5\nD = 0.1\n"
        "[BVV.O]\nW0 = 0.0\nD = 0.0\n"
    )
    p = parse_parameters(str(f), validate=True, species=SPECIES,
                         use_BV=True, use_BVV=True)
    assert list(p.BVV.pairs) == list(p.BV.pairs)

    # It must be an independent copy, not an alias: mutating BVV pairs must
    # not leak into BV pairs (they are fit separately).
    p.BVV.pairs["O-Pb"].r0 = 9.99
    assert p.BV.pairs["O-Pb"].r0 == pytest.approx(2.06)


def test_explicit_bvv_pairs_not_overwritten_by_inheritance(tmp_path):
    f = tmp_path / "explicit.toml"
    f.write_text(
        "cutoff = 6.0\n"
        "[BV.pairs.O-Pb]\nr0 = 2.06\nC = 6.0\n"
        "[BVV.Pb]\nW0 = 0.5\nD = 0.1\n"
        "[BVV.pairs.O-Pb]\nr0 = 1.55\nC = 4.0\n"
    )
    p = parse_parameters(str(f), validate=True, species=SPECIES,
                         use_BV=True, use_BVV=True)
    # Explicit BVV pair wins over the BV-inheritance fallback.
    assert p.BVV.pairs["O-Pb"].r0 == pytest.approx(1.55)
    assert p.BV.pairs["O-Pb"].r0 == pytest.approx(2.06)


# ──────────────────────────────────────────────
# Round-trip
# ──────────────────────────────────────────────

def test_bvv_pairs_survive_save_reload(tmp_path):
    p = generate_parameters(SPECIES, use_coulomb=False, use_repulsive=False,
                            use_BV=False, use_BVV=True, use_angle=False)
    out = tmp_path / "rt.toml"
    save_parameters(p, str(out))
    reloaded = parse_parameters(str(out), validate=True, species=SPECIES,
                                use_BV=False, use_BVV=True)
    assert set(reloaded.BVV.pairs) == set(p.BVV.pairs)
    assert reloaded.BVV.species, "BVV species lost on round-trip"
