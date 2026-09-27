# PbTiO3 Buckingham refit — lambda_reg scan (Phase 0, 2026-07-05)

Replaces the r⁻¹² repulsion (whose fit saturated its bounds: repulsive O-Ti
collapsed to ~1e-32, i.e. no Ti-O short-range wall) with Born-Mayer +
dispersion, and scans the Tikhonov pull toward the physical seed. All runs:
exp BV form, frozen reduced charges, **block split** (temporal tail of each
trajectory held out — the honest generalization metric), log-space magnitude
fitting, seeds in `parameters_seed.toml` (literature Born-Mayer values; they
assume formal charges, so the fit softens them against the reduced charges —
initial loss ~14.8).

## Results (test = held-out temporal tail, 18 frames across 4 temperatures)

| λ_reg | test E (eV/atom) | test F (eV/Å) | genuine bound flags | double well | verdict |
|-------|-----------------|---------------|---------------------|-------------|---------|
| 0     | 0.00538 | **0.305** | 4 (V0 ×3 at bounds, BVV O W0) | **0.6 meV — paraelectric** | **FAIL** |
| 1e-4  | 0.00564 | 0.322 | 4 (Pb V0 low, Ti/O V0 high, BVV Pb W0 high) | 19.1 meV @ 0.18 Å | PASS |
| 1e-3  | 0.00568 | 0.323 | 2 (Ti/O V0 at upper bound) | 19.4 meV @ 0.18 Å | PASS |
| 1e-2  | 0.00591 | 0.326 | 2 (**O-Ti Buckingham A ≈ 0**, Pb S ≈ 0) | 20.9 meV @ 0.18 Å | PASS |
| **1e-1** | 0.00613 | 0.333 | **0** | **21.1 meV @ 0.18 Å** | **PASS** |

(The `buckingham C = 1e-5` and `BVV O ≈ 0` flags present in all runs are the
deliberately-zero seeds sitting at their log floor — intentional absences, not
saturation.)

## Chosen potential: `lam_1e-1/output/fitted_parameters.toml`

The only **all-interior optimum**, and the fitted values are physically
interpretable: V0(Pb)=2.08 / V0(Ti)=4.45 (≈ formal valences 2/4),
r0(O-Pb)=2.00 / r0(O-Ti)=1.94 Å (near Brown-Altermatt 2.112/1.815), healthy
BV stiffnesses (S_O=0.48) that supply a stiff Ti-O short-range wall
(E rises +291 eV when a Ti-O bond is compressed to 1.0 Å, vs +75 eV for the
λ=1e-4 fit). Polar retention on a real 300 K frame: |u| 0.381 → 0.346 Å.
Cost vs λ=1e-4: +3.4% held-out force RMSE — bought: no bound saturation, a
double well, and near-formal valence targets.

λ=0 is the cautionary row: **best force RMSE, wrong physics** (paraelectric,
valence targets destroyed). RMSE alone would have picked it.

Every run directory persists `output/fit_diagnostics.toml` (bound flags) and
`output/ferroelectric_validation.toml` + `double_well.png` (physics verdict).

Reproduce: `cd lam_1e-1 && python ../../../src/main.py`

## Update (2026-07-05, Phase 1-2): smoothing-era refit → `smooth_pruned/` adopted

Phase 1 made the C² cutoff taper (`smooth_width = 1.0`) the default and Phase 2
switched fitting to the analytic parameter Jacobian (6.6× faster, identical
optimum). Refitting `lam_1e-1` under smoothing exposed a structural
redundancy: **O-Ti Buckingham (A, C) and BVV(O) walk to the log floor** — the
BV exponential wall already supplies the Ti-O repulsion, so the optimizer
deletes the duplicate term. Raising λ (3e-1, 1e0) does not prevent the
collapse and only degrades the loss (0.654 / 0.778 vs 0.629).

**Adopted: `smooth_pruned/`** — O-Ti Buckingham pair and BVV(O) species
removed from the model (5 fewer parameters), λ = 1e-1 unchanged:

| model              | loss   | bound flags        | test_F (eV/Å) | well (meV/f.u.) | retention (Å) |
|--------------------|--------|--------------------|---------------|-----------------|----------------|
| full, λ=1e-1       | 0.6289 | 5 (incl. O-Ti A,C) | 0.3348        | 22.0            | 0.346          |
| **pruned, λ=1e-1** | 0.6316 | 1 (O-Pb C → 0)     | 0.3357        | 22.4            | 0.347          |

Same accuracy, all load-bearing parameters interior; the single remaining
flag is the O-Pb dispersion C sitting at its floor (harmless — pure
Born-Mayer O-Pb). `lam_1e-1/output` is kept as the pre-smoothing reference.

## Update 2 (2026-07-05, Phase 4): phantom-bond fix → `smooth_pruned/` refit v2

Cross-validating the LAMMPS export (`scripts/export_lammps.py --validate`)
exposed a real bug: the **exp-form BV/BVV synthesized phantom exp(−r) bonds
on every unparameterized species pair** (Pb-Pb, Pb-Ti, Ti-Ti, O-O) — the
r0=0/b=1 placeholder does not vanish like the power form's (0/r)^C. Fixed
(mask in `_pair_tables`), and the refit got strictly better:

| smooth_pruned      | loss   | test_F (eV/Å) | well (meV/f.u.) | flags |
|--------------------|--------|---------------|-----------------|-------|
| phantom-era        | 0.6316 | 0.3357        | 22.4            | 1     |
| **v2 (fixed)**     | 0.6040 | 0.3237        | 23.7            | 1     |

The v2 output/ is the adopted potential (P_bec = 0.50 C/m², cubic soft mode
−4.25 THz, retention 0.352 Å, all gates PASS). `lammps_export/` next to it
holds the validated LAMMPS files (ΔE 0.016 meV/atom, |ΔF| 1.5e-5 eV/Å vs
Python; 90–400× faster). NOTE: all other exp-form fits in this repo
(lam_* scan points, repo-root output/, PbTiO3_2basin, BaTiO3) predate the
fix and are stale references.
