"""
Ferroelectric validation tools for a fitted BVFF, built on the ASE calculator.

A good *fit* to thermal AIMD forces is necessary but not sufficient — the point
of a ferroelectric potential is that it reproduces the ferroelectric physics:

  - a **double well** in energy vs. polar (cation-sublattice) displacement,
  - a nonzero **spontaneous polarization** in the distorted (ground) state,
  - a relaxed structure that is polar (off-centered), not centrosymmetric.

These helpers compute exactly those observables so a fitted potential can be
judged on physics rather than on force RMSE alone. They are model-agnostic
(any BVFF via BVFFCalculator) and need no DFT.

Run as a script for a quick demo on the fitted PbTiO3 potential:
    python3 scripts/ferroelectric.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script (`python3 scripts/ferroelectric.py`) by putting the
# project root on sys.path before the src/ imports. No-op when imported normally.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.calculator import BVFFCalculator


# 1 e/Å² = 16.0217663 C/m²  (1.602176634e-19 C / (1e-10 m)²)
E_PER_ANG2_TO_C_PER_M2 = 16.0217663

# sqrt(eV / (amu·Å²)) in THz (ν = ω/2π) and the matching ħω in meV.
SQRT_EV_AMU_ANG2_TO_THZ = 15.6330214
SQRT_EV_AMU_ANG2_TO_MEV = 64.654148

# Born effective charges (|e|, cubic phase, from DFT literature: Zhong,
# King-Smith & Vanderbilt PRL 72, 3618 (1994); Ghosez et al.). Keyed by the
# A-site species. O_par is the O whose Ti-O-Ti chain runs along the polar
# axis; O_perp the other two. Each set obeys the acoustic sum rule
# Z*_A + Z*_Ti + Z*_O∥ + 2·Z*_O⊥ = 0. These are what make a *quantitative*
# polarization estimate possible: the reduced FF point charges underestimate
# P by 2-3× because they omit the dynamical (electronic) charge transfer.
ZSTAR_TABLE = {
    "Pb": {"A": 3.90, "Ti": 7.06, "O_par": -5.83, "O_perp": -2.565},
    "Ba": {"A": 2.75, "Ti": 7.16, "O_par": -5.69, "O_perp": -2.11},
}


# ──────────────────────────────────────────────
# Reference structures
# ──────────────────────────────────────────────

def ideal_perovskite(a: float = 3.9, rep=(2, 2, 2), A: str = "Pb", B: str = "Ti"):
    """Ideal **centrosymmetric** cubic ABO3 perovskite (``A`` at the corner,
    ``B`` at the body center, O on faces), repeated to a supercell. This is the
    paraelectric reference whose symmetry the ferroelectric distortion breaks."""
    from ase import Atoms
    basis_scaled = [
        (0.0, 0.0, 0.0),    # A site
        (0.5, 0.5, 0.5),    # B site
        (0.5, 0.5, 0.0),    # O
        (0.5, 0.0, 0.5),    # O
        (0.0, 0.5, 0.5),    # O
    ]
    atoms = Atoms(
        symbols          = [A, B, "O", "O", "O"],
        scaled_positions = basis_scaled,
        cell             = [[a, 0, 0], [0, a, 0], [0, 0, a]],
        pbc              = True,
    )
    return atoms.repeat(rep)


# ──────────────────────────────────────────────
# Relaxation
# ──────────────────────────────────────────────

def relax(atoms, calc, fmax: float = 0.02, steps: int = 500,
          relax_cell: bool = False, logfile=None):
    """Relax atoms (optionally the cell too) with FIRE. Returns the same atoms,
    relaxed in place. The displacement of the relaxed structure from the
    centrosymmetric reference is the ferroelectric order parameter."""
    from ase.optimize import FIRE
    atoms.calc = calc
    target = atoms
    if relax_cell:
        from ase.filters import FrechetCellFilter
        target = FrechetCellFilter(atoms)
    FIRE(target, logfile=logfile).run(fmax=fmax, steps=steps)
    return atoms


# ──────────────────────────────────────────────
# Double-well scan
# ──────────────────────────────────────────────

def double_well_scan(atoms_ref, calc, amplitudes=None, axis: int = 2,
                     move_species=("Ti", "Pb")):
    """
    Energy vs. rigid polar displacement of the cation sublattice.

    Displaces every ``move_species`` atom by δ along ``axis`` (Cartesian, Å)
    from the reference structure, holding O fixed — the prototypical soft-mode
    coordinate. Returns ``(amplitudes, dE_per_fu)`` with energy measured per
    formula unit relative to δ = 0. A ferroelectric potential shows a **double
    well**: minima at δ ≠ 0 below the δ = 0 energy.
    """
    if amplitudes is None:
        amplitudes = np.linspace(-0.5, 0.5, 51)
    sym  = np.array(atoms_ref.get_chemical_symbols())
    mask = np.isin(sym, list(move_species))
    # formula units = number of B-site cations (Ti) — 1 per ABO3.
    n_fu = max(1, int(np.sum(sym == "Ti")))

    energies = np.empty(len(amplitudes))
    base_pos = atoms_ref.get_positions()
    for k, d in enumerate(amplitudes):
        a = atoms_ref.copy()
        pos = base_pos.copy()
        pos[mask, axis] += d
        a.set_positions(pos)
        a.calc = calc
        energies[k] = a.get_potential_energy()

    i0 = int(np.argmin(np.abs(amplitudes)))     # δ ≈ 0
    return np.asarray(amplitudes), (energies - energies[i0]) / n_fu


def well_depth(amplitudes, dE_per_fu):
    """Summarize a double-well scan: ``(depth_meV, delta_min)``. ``depth`` is the
    energy gain (meV/f.u.) of the deepest off-center minimum relative to δ=0
    (positive ⇒ a real double well); ``delta_min`` is its displacement (Å)."""
    off = np.abs(amplitudes) > 1e-9
    if not np.any(off):
        return 0.0, 0.0
    kmin = np.argmin(np.where(off, dE_per_fu, np.inf))
    return float(-dE_per_fu[kmin] * 1000.0), float(amplitudes[kmin])


# ──────────────────────────────────────────────
# Spontaneous polarization (point-charge)
# ──────────────────────────────────────────────

def cation_offcentering(atoms, cation: str = "Ti", n_neighbors: int = 6):
    """Mean off-centering vector (Å) of a cation sublattice from its O cage — an
    origin-independent ferroelectric order parameter. For each ``cation`` atom,
    take the displacement to the centroid of its ``n_neighbors`` nearest oxygens
    (minimum image); the cation off-centering is minus that centroid. A polar
    (ferroelectric) structure gives a large coherent vector; a centrosymmetric
    one gives ≈ 0. This is the *right* quantity to validate against, computed on
    real (polar) training structures rather than an idealized cubic reference."""
    from itertools import product
    L    = np.asarray(atoms.cell.array, dtype=float)
    inv  = np.linalg.inv(L)
    cart = atoms.get_positions()
    sym  = np.array(atoms.get_chemical_symbols())
    O    = np.where(sym == "O")[0]
    cats = np.where(sym == cation)[0]
    if O.size == 0 or cats.size == 0:
        return np.zeros(3)
    # A small cell can need several periodic images of the SAME O atom among
    # the n nearest neighbors (the 5-atom primitive perovskite has 3 O atoms
    # but a 6-O octahedron), so expand each minimum-image O over the 27
    # adjacent images before ranking — with only the single nearest image a
    # centrosymmetric primitive cell reports a spurious |u| = a·√3/6.
    shifts = np.array(list(product((-1, 0, 1), repeat=3)), dtype=float)
    vecs = []
    for i in cats:
        df = (cart[O] - cart[i]) @ inv
        df -= np.round(df)                            # nearest image of each O
        d  = (df[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ L
        nn = np.argsort(np.linalg.norm(d, axis=1))[:n_neighbors]
        vecs.append(-d[nn].mean(axis=0))
    return np.asarray(vecs).mean(axis=0)


def polarization(atoms, charges: dict, atoms_ref):
    """
    Point-charge spontaneous polarization (C/m²):

        P = (1/V) Σ_i q_i u_i,    u_i = minimum-image(r_i − r_i^ref)

    measured relative to the centrosymmetric reference ``atoms_ref`` (same atom
    order). Displacements are taken in fractional coordinates with a
    minimum-image wrap (df -= round(df)): an atom that crossed a periodic
    boundary during relax/MD would otherwise contribute a spurious lattice-
    vector jump of ~q·L/V. As in Berry-phase theory, P is only defined modulo
    the polarization quantum (e·L_axis/V per boundary crossing); this picks the
    branch nearest the reference, valid while true displacements stay < L/2.
    This is the rigid-ion estimate of the ferroelectric order parameter;
    it is exact for a rigid-ion model and a useful proxy otherwise (a Berry-phase
    treatment would be needed for the true electronic contribution).
    """
    sym = atoms.get_chemical_symbols()
    q   = np.array([charges.get(s, 0.0) for s in sym])
    L   = np.asarray(atoms.cell.array, dtype=float)
    df  = atoms.get_scaled_positions(wrap=False) - atoms_ref.get_scaled_positions(wrap=False)
    df -= np.round(df)                                         # minimum image
    u   = df @ L                                               # (N,3) Å
    V   = atoms.get_volume()                                   # Å³
    P_e_per_ang2 = (q[:, None] * u).sum(axis=0) / V            # e/Å²
    return P_e_per_ang2 * E_PER_ANG2_TO_C_PER_M2               # C/m²


def perovskite_zstar(atoms_ref, axis: int = 2, A_site: str | None = None,
                     table: dict | None = None) -> np.ndarray | None:
    """
    Per-atom scalar Born effective charges for an ATiO3 perovskite, for
    polarization along ``axis``.

    Each O is classified by its Ti-O-Ti chain direction (the min-image vector
    to its nearest Ti — robust for distorted cells): chains along ``axis`` get
    Z*_O∥, the others Z*_O⊥. A and Ti sites get their scalar Z*. Returns an
    (N,) array, or None when the chemistry is not in ``ZSTAR_TABLE``.
    """
    sym = np.array(atoms_ref.get_chemical_symbols())
    if A_site is None:
        others = [s for s in dict.fromkeys(sym.tolist()) if s not in ("Ti", "O")]
        A_site = others[0] if others else ""
    tab = (table or ZSTAR_TABLE).get(A_site)
    if tab is None or "Ti" not in sym:
        return None

    L    = np.asarray(atoms_ref.cell.array, dtype=float)
    inv  = np.linalg.inv(L)
    cart = atoms_ref.get_positions()
    Ti   = np.where(sym == "Ti")[0]
    z    = np.zeros(len(sym))
    z[sym == A_site] = tab["A"]
    z[sym == "Ti"]   = tab["Ti"]
    for i in np.where(sym == "O")[0]:
        df = (cart[Ti] - cart[i]) @ inv
        df -= np.round(df)
        d  = df @ L
        chain = int(np.argmax(np.abs(d[np.argmin(np.linalg.norm(d, axis=1))])))
        z[i] = tab["O_par"] if chain == axis else tab["O_perp"]
    return z


def polarization_bec(atoms, atoms_ref, zstar: np.ndarray) -> np.ndarray:
    """
    Born-effective-charge polarization (C/m²): P = (1/V) Σ_i Z*_i u_i with the
    same minimum-image displacement convention as ``polarization``. With
    literature Z* this is quantitatively comparable to experiment/DFT
    (PbTiO3 P_s ≈ 0.75 C/m², BaTiO3 ≈ 0.26), unlike the reduced-FF-charge
    estimate which underestimates by the missing dynamical charge transfer.
    Valid to first order in the displacements from ``atoms_ref``.
    """
    zstar = np.asarray(zstar, dtype=float)
    L  = np.asarray(atoms.cell.array, dtype=float)
    df = atoms.get_scaled_positions(wrap=False) - atoms_ref.get_scaled_positions(wrap=False)
    df -= np.round(df)
    u  = df @ L
    V  = atoms.get_volume()
    return (zstar[:, None] * u).sum(axis=0) / V * E_PER_ANG2_TO_C_PER_M2


# ──────────────────────────────────────────────
# Γ-point phonons
# ──────────────────────────────────────────────

def gamma_phonons(atoms, calc, dx: float = 0.01):
    """
    Γ-point phonon frequencies from a central-difference Hessian of the
    (analytic) forces: H[3i+a, 3j+b] = −∂F_{jb}/∂x_{ia}, mass-weighted and
    diagonalized. Imaginary modes are returned as *negative* frequencies.

    Returns ``(freqs_THz, freqs_meV, n_imaginary)`` with frequencies sorted
    ascending; the 3 acoustic modes sit at ≈ 0. The physics check this
    enables: a ferroelectric's **cubic** phase must show an unstable (soft,
    imaginary) polar mode at Γ, while the **relaxed polar** phase should have
    none — the soft mode is the microscopic mechanism behind the double well.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    n   = len(atoms)
    pos = atoms.get_positions()
    H   = np.zeros((3 * n, 3 * n))
    for i in range(n):
        for a in range(3):
            for s, sign in ((0, +1.0), (1, -1.0)):
                p = pos.copy()
                p[i, a] += sign * dx
                atoms.set_positions(p)
                f = atoms.get_forces().ravel()
                H[3 * i + a] += -sign * f / (2.0 * dx)
    atoms.set_positions(pos)
    H = 0.5 * (H + H.T)

    m     = np.repeat(atoms.get_masses(), 3)
    D     = H / np.sqrt(np.outer(m, m))
    evals = np.linalg.eigvalsh(D)
    # Zero out acoustic-scale numerical noise so tiny negative eigenvalues of
    # the translational modes don't masquerade as soft modes.
    tol   = 1e-4 * max(1.0, np.abs(evals).max())
    n_im  = int(np.sum(evals < -tol))
    freqs = np.sign(evals) * np.sqrt(np.abs(evals))
    return (freqs * SQRT_EV_AMU_ANG2_TO_THZ,
            freqs * SQRT_EV_AMU_ANG2_TO_MEV,
            n_im)


# ──────────────────────────────────────────────
# Molecular dynamics (NVT)
# ──────────────────────────────────────────────

def run_nvt(atoms, calc, temperature_K: float, steps: int = 1000,
            timestep_fs: float = 1.0, friction: float = 0.01, seed: int = 0,
            logfile=None):
    """Short Langevin NVT run — confirms the potential drives stable dynamics.
    Returns the trajectory's potential energies (eV). Uses a fixed RNG seed for
    reproducibility."""
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units

    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed))
    dyn = Langevin(atoms, timestep_fs * units.fs, temperature_K=temperature_K,
                   friction=friction, logfile=logfile, rng=np.random.default_rng(seed))
    energies = []
    def _record():
        energies.append(atoms.get_potential_energy())
    dyn.attach(_record, interval=max(1, steps // 50))
    dyn.run(steps)
    return np.asarray(energies)


# ──────────────────────────────────────────────
# Persisted validation gate (called from src/main.py after every fit; also the
# script entry point). The ABO3 chemistry is detected from a training frame —
# works for PbTiO3, BaTiO3, ...
# ──────────────────────────────────────────────

def _detect_chemistry(data_frame=None):
    """A-site species and pseudo-cubic lattice constant from a training frame
    (5-atom-f.u. ABO3 assumed: A = the species that is neither Ti nor O).
    PbTiO3 defaults when no frame is given."""
    A_site, a_ref = "Pb", 3.97
    if data_frame is not None:
        others = [s for s in dict.fromkeys(data_frame.species) if s not in ("Ti", "O")]
        if others:
            A_site = others[0]
        n_fu  = max(1, len(data_frame.species) // 5)
        vol   = abs(np.linalg.det(np.asarray(data_frame.lattice, dtype=float)))
        a_ref = float((vol / n_fu) ** (1.0 / 3.0))
    return A_site, a_ref


def run_validation(
    calc,
    output_dir,
    charges:           dict | None = None,
    data_frame                     = None,
    amplitudes                     = None,
    relax_steps:       int         = 300,
    nvt_steps:         int         = 200,
    nvt_temperature_K: float       = 300.0,
    logger                         = None,
) -> dict:
    """
    Run the ferroelectric acceptance tests on a fitted potential and PERSIST
    the verdict — RMSE alone cannot tell whether a fit is ferroelectric, and
    an unrecorded stdout demo cannot be audited after the fact.

    Tests (thresholds documented inline):
      1. **Double well**: rigid polar cation displacement scan on the ideal
         cubic reference. A genuine ferroelectric well has an *interior*
         minimum at modest δ and shallow-ish depth; a minimum at the scan edge
         / very deep / at large δ is cation-O collapse, not ferroelectricity.
      2. **Point-charge P** at the well minimum (when ``charges`` given) —
         rigid-ion estimate with the (reduced) FF charges, systematically
         below Born-effective-charge values; recorded for trend tracking, not
         used in the pass/fail gate.
      3. **Polar retention** on a real ``data_frame`` (authoritative test):
         relax a polar training configuration and require the Ti off-centering
         to survive, judged relative to its starting value (PbTiO3 |u|~0.38 Å
         vs BaTiO3 ~0.1 Å — a fixed absolute threshold would misjudge soft
         ferroelectrics).
      4. **NVT stability**: short Langevin run stays finite.

    Writes ``ferroelectric_validation.toml`` and ``double_well.png`` (when
    matplotlib is available) into ``output_dir``. Returns a dict mirroring the
    TOML; ``result["passed"]`` is the overall gate.
    """
    from pathlib import Path
    log = logger.info if logger is not None else (lambda msg: None)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    A_site, a_ref = _detect_chemistry(data_frame)
    log(f"Chemistry: {A_site}TiO3 | pseudo-cubic a = {a_ref:.3f} Å"
        + (" (from training cell)" if data_frame is not None else " (defaults)"))
    ref = ideal_perovskite(a=a_ref, rep=(2, 2, 2), A=A_site)   # centrosymmetric reference

    # 1. Double-well scan.
    log(f"── Double-well scan (rigid Ti+{A_site} displacement along z) ──")
    amps, dE = double_well_scan(ref, calc, amplitudes=amplitudes,
                                move_species=("Ti", A_site))
    depth, dmin = well_depth(amps, dE)
    if len(amps) > 1:
        step    = float(np.max(np.diff(np.sort(amps))))   # robust to non-uniform grids
        at_edge = abs(abs(dmin) - float(np.max(np.abs(amps)))) < step + 1e-9
    else:
        at_edge = True                       # a 1-point scan cannot show an interior minimum
    if not np.isfinite(dE).all():
        classification = "invalid"           # NaN/Inf on the scan: fail loudly, never pass
    elif depth <= 1.0:
        classification = "paraelectric"      # single minimum at δ=0
    elif at_edge or abs(dmin) > 0.35 or depth > 800.0:
        classification = "collapse"          # runaway toward O, not a bound well
    else:
        classification = "double_well"       # interior minimum: ferroelectric-like
    log(f"  deepest off-center min: {depth:+.1f} meV/f.u. at δ={dmin:+.3f} Å "
        f"→ {classification}")

    # 2. Polarization at the well minimum (same atom order as ref, so the
    #    displacement branch is unambiguous): point-charge estimate with the
    #    FF charges AND — when the chemistry has tabulated Born effective
    #    charges — the quantitative BEC estimate.
    P_min, P_bec = None, None
    if classification != "paraelectric":
        a   = ref.copy()
        pos = a.get_positions()
        sym = np.array(a.get_chemical_symbols())
        pos[np.isin(sym, ["Ti", A_site]), 2] += dmin
        a.set_positions(pos)
        if charges:
            P_min = polarization(a, charges, ref)
            log(f"  point-charge P at δ_min: ({P_min[0]:+.3f}, {P_min[1]:+.3f}, "
                f"{P_min[2]:+.3f}) C/m² (rigid-ion, reduced FF charges)")
        zstar = perovskite_zstar(ref, axis=2, A_site=A_site)
        if zstar is not None:
            P_bec = polarization_bec(a, ref, zstar)
            log(f"  BEC P at δ_min:          ({P_bec[0]:+.3f}, {P_bec[1]:+.3f}, "
                f"{P_bec[2]:+.3f}) C/m² (literature Z*, quantitative estimate)")

    # 3. Polar-state retention on a REAL data frame.
    retention: dict = {"tested": False}
    if data_frame is not None:
        from src.calculator import atoms_from_frame
        a = atoms_from_frame(data_frame)
        a.calc = calc
        u0 = float(np.linalg.norm(cation_offcentering(a, "Ti")))
        if u0 < 0.05:
            # A frame that starts with no measurable off-centering (e.g. a
            # near-cubic high-T snapshot) cannot test retention: u1 > 0.05
            # would fail even a potential that fully preserves polarity.
            retention = {"tested": False, "u0_ang": u0,
                         "note": "starting frame not measurably polar (|u0| < 0.05 Å)"}
            log(f"  (retention test skipped — starting |u| = {u0:.3f} Å is not "
                f"measurably polar; pass a polar frame)")
        else:
            relax(a, calc, fmax=0.05, steps=relax_steps)
            u1 = float(np.linalg.norm(cation_offcentering(a, "Ti")))
            retained  = bool(u1 > max(0.05, 0.5 * u0))
            retention = {"tested": True, "u0_ang": u0, "u1_ang": u1, "retained": retained}
            log(f"  Ti off-centering |u|: {u0:.3f} → {u1:.3f} Å after relax → "
                + ("polar state preserved" if retained else "POLARITY LOST"))
    else:
        log("  (retention test skipped — no data frame available)")

    # 3b. Γ-phonons of the cubic primitive cell: a ferroelectric must show an
    # unstable (imaginary) polar mode — the soft-mode mechanism behind the
    # double well. Recorded, not gated (the double-well scan already gates the
    # same physics through a bigger displacement). Guarded: a blown-up
    # parameter set can NaN the Hessian, and a diagnostic must never kill the
    # validation that reports it.
    try:
        prim = ideal_perovskite(a=a_ref, rep=(1, 1, 1), A=A_site)
        f_thz, f_mev, n_im = gamma_phonons(prim, calc)
        log(f"  Γ-phonons (cubic primitive): {n_im} unstable mode(s); "
            f"softest = {f_thz[0]:+.2f} THz ({f_mev[0]:+.1f} meV)"
            + (" → soft mode present" if n_im else " → NO soft mode (paraelectric-like)"))
        phonons = {
            "n_unstable_modes": int(n_im),
            "softest_THz":      float(f_thz[0]),
            "softest_meV":      float(f_mev[0]),
            "note": "cubic primitive cell at Γ; a ferroelectric shows >= 1 "
                    "imaginary (negative) polar mode — recorded, not gated",
        }
    except Exception as exc:
        log(f"  Γ-phonons skipped ({exc})")
        phonons = {"note": f"skipped: {exc}"}

    # 4. Short NVT MD stability check.
    e = run_nvt(ref.copy(), calc, temperature_K=nvt_temperature_K, steps=nvt_steps)
    drift    = float(np.max(np.abs(e - e[0]))) if e.size else float("nan")
    exploded = bool((not np.isfinite(e).all()) or drift > 1.0e3)
    log(f"  {nvt_temperature_K:.0f} K Langevin, {nvt_steps} steps: "
        f"|ΔE|max = {drift:.3g} eV → " + ("UNSTABLE" if exploded else "stable"))

    reasons = []
    if classification != "double_well":
        reasons.append(f"double-well scan classified as '{classification}'")
    if retention.get("tested") and not retention.get("retained"):
        reasons.append("polar off-centering lost on relaxation of a training frame")
    if exploded:
        reasons.append("NVT dynamics unstable / non-finite")
    passed = not reasons

    result: dict = {
        "passed":    passed,
        "reasons":   reasons,
        "chemistry": {"A_site": A_site, "a_ref_ang": float(a_ref)},
        "double_well": {
            "classification":   classification,
            "depth_meV_per_fu": float(depth),
            "delta_min_ang":    float(dmin),
            "at_scan_edge":     bool(at_edge),
            "amplitudes_ang":   [float(v) for v in amps],
            "dE_meV_per_fu":    [float(v) * 1000.0 for v in dE],
        },
        "retention": retention,
        "gamma_phonons_cubic": phonons,
        "nvt": {
            "temperature_K":    float(nvt_temperature_K),
            "steps":            int(nvt_steps),
            "max_abs_drift_eV": drift,
            "exploded":         exploded,
        },
    }
    if P_min is not None or P_bec is not None:
        result["polarization"] = {}
        if P_min is not None:
            result["polarization"].update({
                "P_at_delta_min_C_per_m2": [float(v) for v in P_min],
                "note": "rigid-point-charge estimate with the reduced FF charges; "
                        "systematically below Born-effective-charge values",
            })
        if P_bec is not None:
            result["polarization"].update({
                "P_bec_at_delta_min_C_per_m2": [float(v) for v in P_bec],
                "bec_note": "literature Born effective charges (ZSTAR_TABLE); "
                            "quantitative first-order estimate — compare "
                            "PbTiO3 P_s ~ 0.75 C/m2, BaTiO3 ~ 0.26 C/m2",
            })

    import tomli_w
    with open(out / "ferroelectric_validation.toml", "wb") as f:
        tomli_w.dump(result, f)

    # The TOML above is the record of truth; the plot is a convenience. Never
    # let a plotting problem (missing matplotlib, backend conflict with an
    # earlier pyplot import in the same run) fail the validation itself.
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        ax.plot(amps, np.asarray(dE) * 1000.0, "o-", ms=3)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xlabel(f"δ (Å) — rigid Ti+{A_site} displacement")
        ax.set_ylabel("ΔE (meV/f.u.)")
        ax.set_title(f"{classification}: depth {depth:.1f} meV/f.u. at δ={dmin:+.3f} Å")
        fig.tight_layout()
        fig.savefig(out / "double_well.png", dpi=150)
        plt.close(fig)
    except Exception as exc:                                   # pragma: no cover
        log(f"  (double_well.png skipped: {exc})")

    log(f"Ferroelectric validation: {'PASS' if passed else 'FAIL'}"
        + (f" ({'; '.join(reasons)})" if reasons else ""))
    return result


def _demo() -> int:
    """Script entry point: validate the fitted potential of the current run
    directory (controls.toml + output/fitted_parameters.toml) and persist the
    verdict next to it. Exits nonzero on a failed validation, so this can gate
    a refit pipeline."""
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("bvff.fe")

    from parsers.controls_parser   import parse_controls
    from parsers.parameters_parser import parse_parameters
    from src.main                  import build_bvff

    controls = parse_controls("controls.toml")
    fitted   = parse_parameters("output/fitted_parameters.toml", validate=True)

    data_frame = None
    try:
        from parsers.dataset import load_dataset
        data_frame = load_dataset(entries=controls.dataset).frames[0]
    except Exception as exc:                                  # pragma: no cover
        log.info(f"  (no data frame — retention test will be skipped: {exc})")

    calc   = BVFFCalculator(build_bvff(controls, fitted))
    result = run_validation(
        calc,
        output_dir = controls.output_dir,
        charges    = fitted.coulomb.charges,
        data_frame = data_frame,
        logger     = log,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
