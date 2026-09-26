from __future__ import annotations

import tomllib
import tomli_w
from pathlib import Path
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

from .defaults import DEFAULT_PARAMETERS


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────

@dataclass
class CoulombParams:
    charges: dict[str, float] = field(default_factory=dict)


@dataclass
class RepulsiveParams:
    B: dict[str, float] = field(default_factory=dict)


@dataclass
class BuckinghamPair:
    A:   float = 1.0
    rho: float = 0.3
    C:   float = 0.0


@dataclass
class BuckinghamParams:
    pairs: dict[str, BuckinghamPair] = field(default_factory=dict)


@dataclass
class BVSpecies:
    V0: float = 1.0
    S:  float = 1.0


@dataclass
class BVPair:
    r0: float = 1.0
    C:  float = 1.0    # power-law exponent (form="power")
    b:  float = 0.37   # Brown-Altermatt decay length in Å (form="exp")


@dataclass
class BVParams:
    species: dict[str, BVSpecies] = field(default_factory=dict)
    pairs:   dict[str, BVPair]   = field(default_factory=dict)


@dataclass
class BVVSpecies:
    W0: float = 1.0
    D:  float = 1.0


@dataclass
class BVVParams:
    species: dict[str, BVVSpecies] = field(default_factory=dict)


@dataclass
class AngleParams:
    k: float = 1.0


@dataclass
class Parameters:
    cutoff:     float            = DEFAULT_PARAMETERS["cutoff"]
    cutoff_width: float          = DEFAULT_PARAMETERS["cutoff_width"]
    coulomb:    CoulombParams    = field(default_factory=CoulombParams)
    repulsive:  RepulsiveParams  = field(default_factory=RepulsiveParams)
    buckingham: BuckinghamParams = field(default_factory=BuckinghamParams)
    BV:         BVParams         = field(default_factory=BVParams)
    BVV:        BVVParams        = field(default_factory=BVVParams)
    angle:      AngleParams      = field(default_factory=AngleParams)
    # Per-species reference energy μ_s (eV/atom), E_ref = Σ n_s μ_s. Written
    # by the fit (least-squares, see fitting.fit_energy_reference); optional
    # on input. Not a fitted SA parameter.
    energy_ref: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Auto-generation
# ──────────────────────────────────────────────

def _generate_pairs(species: list[str]) -> list[str]:
    """Generate all unique pairs from species list."""
    return [f"{a}-{b}" for a, b in combinations_with_replacement(sorted(species), 2)]


# Common anions in inorganic systems. Used to skip physically-meaningless
# cation–cation pairs in BV (bond valence is defined only for cation→anion
# bonds). If none of the system's species are in this set, we fall back to
# emitting all pairs and warn — the user can then prune by hand.
_COMMON_ANIONS = frozenset({"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "P", "H"})


def _generate_bv_pairs(species: list[str]) -> list[str]:
    """Generate BV pairs that include at least one common anion. Skip
    cation–cation pairs because bond valence is undefined for them.
    """
    species_sorted = sorted(species)
    anions = [s for s in species_sorted if s in _COMMON_ANIONS]
    if not anions:
        # No recognised anion → fall back to all pairs; user must prune.
        return _generate_pairs(species)
    out = []
    for a, b in combinations_with_replacement(species_sorted, 2):
        if a in _COMMON_ANIONS or b in _COMMON_ANIONS:
            out.append(f"{a}-{b}")
    return out


def generate_parameters(
    species:        list[str],
    use_coulomb:    bool,
    use_repulsive:  bool,
    use_BV:         bool,
    use_BVV:        bool,
    use_angle:      bool,
    use_buckingham: bool = False,
) -> Parameters:
    """
    Auto-generate Parameters with all values set to 1.0 based on species
    and enabled potentials.

    Notes
    -----
    - Coulomb charges and Repulsive B are emitted for every pair, since
      both have a sensible value (charge / 0) for any combination.
    - BV pairs are restricted to combinations that include a common anion
      (O, F, Cl, …): bond valence is defined for cation→anion bonds only.
      If no recognised anion is present, all pairs are emitted as a fallback
      and the user is expected to prune by hand.

    Args:
        species:       list of unique element symbols from vasprun.xml
        use_*:         potential toggles from controls.toml

    Returns:
        Parameters dataclass with placeholder values = 1.0.
    """
    all_pairs = _generate_pairs(species)
    bv_pairs  = _generate_bv_pairs(species)
    params    = Parameters()

    if use_coulomb:
        params.coulomb = CoulombParams(
            charges={s: 1.0 for s in species}
        )

    if use_repulsive:
        params.repulsive = RepulsiveParams(
            B={pair: 1.0 for pair in all_pairs}
        )

    if use_buckingham:
        params.buckingham = BuckinghamParams(
            pairs={pair: BuckinghamPair(A=1.0, rho=0.3, C=0.0) for pair in all_pairs}
        )

    if use_BV:
        params.BV = BVParams(
            species={s: BVSpecies(V0=1.0, S=1.0) for s in species},
            pairs={pair: BVPair(r0=1.0, C=1.0) for pair in bv_pairs},
        )

    if use_BVV:
        params.BVV = BVVParams(
            species={s: BVVSpecies(W0=1.0, D=1.0) for s in species}
        )

    if use_angle:
        params.angle = AngleParams(k=1.0)

    return params


def save_parameters(params: Parameters, filepath: str) -> None:
    """
    Save Parameters dataclass to a TOML file.

    Args:
        params:   Parameters dataclass
        filepath: output path for parameters.toml
    """
    data: dict = {"cutoff": params.cutoff, "cutoff_width": params.cutoff_width}

    if params.coulomb.charges:
        data["coulomb"] = {atom: q for atom, q in params.coulomb.charges.items()}

    if params.repulsive.B:
        data["repulsive"] = {pair: b for pair, b in params.repulsive.B.items()}

    if params.buckingham.pairs:
        data["buckingham"] = {
            pair: {"A": bp.A, "rho": bp.rho, "C": bp.C}
            for pair, bp in params.buckingham.pairs.items()
        }

    if params.BV.species or params.BV.pairs:
        data["BV"] = {}
        if params.BV.species:
            data["BV"]["species"] = {
                atom: {"V0": sp.V0, "S": sp.S}
                for atom, sp in params.BV.species.items()
            }
        if params.BV.pairs:
            data["BV"]["pairs"] = {
                pair: {"r0": p.r0, "C": p.C, "b": p.b}
                for pair, p in params.BV.pairs.items()
            }

    if params.BVV.species:
        data["BVV"] = {
            atom: {"W0": sp.W0, "D": sp.D}
            for atom, sp in params.BVV.species.items()
        }

    if params.angle.k != 0.0:
        data["angle"] = {"k": params.angle.k}

    if params.energy_ref:
        data["energy_ref"] = {s: float(mu) for s, mu in params.energy_ref.items()}

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        tomli_w.dump(data, f)


# ──────────────────────────────────────────────
# Parsers
# ──────────────────────────────────────────────

def _parse_coulomb(data: dict) -> CoulombParams:
    raw = data.get("coulomb", DEFAULT_PARAMETERS["coulomb"])
    return CoulombParams(
        charges={atom: float(charge) for atom, charge in raw.items()}
    )


def _parse_repulsive(data: dict) -> RepulsiveParams:
    raw = data.get("repulsive", DEFAULT_PARAMETERS["repulsive"])
    return RepulsiveParams(
        B={pair: float(b) for pair, b in raw.items()}
    )


def _parse_buckingham(data: dict) -> BuckinghamParams:
    raw = data.get("buckingham", DEFAULT_PARAMETERS["buckingham"])
    return BuckinghamParams(
        pairs={
            pair: BuckinghamPair(
                A=float(v["A"]), rho=float(v["rho"]), C=float(v.get("C", 0.0))
            )
            for pair, v in raw.items()
        }
    )


def _parse_BV(data: dict) -> BVParams:
    raw = data.get("BV", DEFAULT_PARAMETERS["BV"])
    species = {
        atom: BVSpecies(V0=float(v["V0"]), S=float(v["S"]))
        for atom, v in raw.get("species", DEFAULT_PARAMETERS["BV"]["species"]).items()
    }
    pairs = {
        pair: BVPair(r0=float(v["r0"]), C=float(v["C"]), b=float(v.get("b", 0.37)))
        for pair, v in raw.get("pairs", DEFAULT_PARAMETERS["BV"]["pairs"]).items()
    }
    return BVParams(species=species, pairs=pairs)


def _parse_BVV(data: dict) -> BVVParams:
    raw = data.get("BVV", DEFAULT_PARAMETERS["BVV"])
    species = {
        atom: BVVSpecies(W0=float(v["W0"]), D=float(v["D"]))
        for atom, v in raw.items()
    }
    return BVVParams(species=species)


def _parse_angle(data: dict) -> AngleParams:
    raw = data.get("angle", DEFAULT_PARAMETERS["angle"])
    return AngleParams(k=float(raw.get("k", DEFAULT_PARAMETERS["angle"]["k"])))


def _validate(params: Parameters) -> None:
    if params.cutoff <= 0:
        raise ValueError(f"cutoff must be > 0, got {params.cutoff}.")
    if not (0.0 <= params.cutoff_width < params.cutoff):
        raise ValueError(
            f"cutoff_width must satisfy 0 <= cutoff_width < cutoff, got {params.cutoff_width}."
        )

    for pair, b in params.repulsive.B.items():
        if b < 0:
            raise ValueError(f"Repulsive B for pair '{pair}' must be >= 0, got {b}.")

    for pair, bp in params.buckingham.pairs.items():
        if bp.A < 0:
            raise ValueError(f"Buckingham A for pair '{pair}' must be >= 0, got {bp.A}.")
        if bp.rho <= 0:
            raise ValueError(f"Buckingham rho for pair '{pair}' must be > 0, got {bp.rho}.")
        if bp.C < 0:
            raise ValueError(f"Buckingham C for pair '{pair}' must be >= 0, got {bp.C}.")

    for atom, sp in params.BV.species.items():
        if sp.S < 0:
            raise ValueError(f"BV S for atom '{atom}' must be >= 0, got {sp.S}.")

    for pair, p in params.BV.pairs.items():
        if p.b <= 0:
            raise ValueError(f"BV b for pair '{pair}' must be > 0, got {p.b}.")

    if params.angle.k < 0:
        raise ValueError(f"Angle spring constant k must be >= 0, got {params.angle.k}.")


def parse_parameters(
    filepath:       str  = "parameters.toml",
    validate:       bool = True,
    species:        list[str] | None = None,
    use_coulomb:    bool = True,
    use_repulsive:  bool = True,
    use_BV:         bool = True,
    use_BVV:        bool = True,
    use_angle:      bool = False,
    use_buckingham: bool = False,
    output_dir:     str  = "./output",
) -> Parameters:
    """
    Parse parameters.toml. If file does not exist, auto-generate with all values = 1.0
    based on species from vasprun.xml and enabled potentials from controls.toml.

    Args:
        filepath:      path to parameters.toml
        validate:      if True, validate parsed values
        species:       unique element symbols from vasprun.xml (required for auto-generation)
        use_*:         potential toggles from controls.toml
        output_dir:    directory to save auto-generated parameters.toml

    Returns:
        Parameters dataclass
    """
    path = Path(filepath)

    # Auto-generate if file does not exist
    if not path.exists():
        if species is None:
            raise ValueError(
                "parameters.toml not found and no species provided for auto-generation. "
                "Please provide species from vasprun.xml."
            )
        params = generate_parameters(
            species        = species,
            use_coulomb    = use_coulomb,
            use_repulsive  = use_repulsive,
            use_BV         = use_BV,
            use_BVV        = use_BVV,
            use_angle      = use_angle,
            use_buckingham = use_buckingham,
        )
        save_path = str(Path(output_dir) / "parameters.toml")
        save_parameters(params, save_path)
        return params

    # Parse existing file
    with open(path, "rb") as f:
        data = tomllib.load(f)

    d = DEFAULT_PARAMETERS
    params = Parameters(
        cutoff     = float(data.get("cutoff", d["cutoff"])),
        cutoff_width = float(data.get("cutoff_width", d["cutoff_width"])),
        coulomb    = _parse_coulomb(data),
        repulsive  = _parse_repulsive(data),
        buckingham = _parse_buckingham(data),
        BV         = _parse_BV(data),
        BVV        = _parse_BVV(data),
        angle      = _parse_angle(data),
        energy_ref = {str(k): float(v) for k, v in data.get("energy_ref", {}).items()},
    )

    if validate:
        _validate(params)

    return params


if __name__ == "__main__":
    # Auto-generation example
    params = parse_parameters(
        filepath      = "parameters.toml",
        validate      = False,
        species       = ["Pb", "Ti", "O"],
        use_coulomb   = True,
        use_repulsive = True,
        use_BV        = True,
        use_BVV       = True,
        use_angle     = False,
    )
    print(params)
