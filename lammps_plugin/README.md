# LAMMPS export of the BVFF

A fitted BVFF runs in mainline LAMMPS with **one** custom style:

| BVFF term            | LAMMPS side                                          |
|----------------------|------------------------------------------------------|
| BV (valence sum)     | `pair_style eam/fs` — exact: F(ρ)=S(ρ−V0)², ρ_ij=V_ij |
| Buckingham (+ r⁻¹²)  | the eam/fs pair table φ(r), guard + C² taper baked in |
| Coulomb (Ewald)      | native `coul/long` + `kspace_style ewald` (tinfoil)   |
| BVV (vector sum)     | **`pair_style bvv`** — this plugin (not expressible in mainline) |
| Angle                | not exported (error if enabled)                       |

## Build

Needs a LAMMPS built with `-DPKG_KSPACE=on -DPKG_MANYBODY=on -DPKG_PLUGIN=on`
(MANYBODY provides eam/fs; PLUGIN provides `plugin load`). Then:

```sh
g++ -std=c++17 -O2 -shared -fPIC \
    -I<lammps>/src -I<lammps-build>/styles -I<lammps-build>/includes/lammps \
    pair_bvv.cpp bvvplugin.cpp -o bvvplugin.so
```

## Use

From a fit directory (`controls.toml` + `output/fitted_parameters.toml`):

```sh
bvff-export-lammps                 # writes lammps_export/
bvff-export-lammps --validate --lmp /path/to/lmp
```

`lammps_export/in.bvff` is a ready single-point input; extend with
`fix nve` / `fix npt` etc. for production MD.

## Validation status (PbTiO3 smooth_pruned, 2026-07-05)

Single-point vs the Python reference on a 40-atom training frame:
ΔE = 0.016 meV/atom, |ΔF|max = 1.5e-5 eV/Å (residual = LAMMPS qqr2e
14.399645 vs Python 14.3996 and kspace accuracy). Throughput (1 CPU proc,
50-step NVE): 162×/409×/90× faster than the Python engine at 40/135/320
atoms; ~350× the NequIP ML-IAP CPU baseline at 40 atoms.

This cross-validation is what exposed the exp-form phantom-bond bug
(unparameterized pairs contributing exp(−r); fixed in bvff/core/potentials.py the
same day) — keep running `--validate` after every refit.

## pair_style bvv parameter file

Written by the exporter (`bvff.bvv`); hand-editable:

```
cutoff        6.0
smooth_width  1.0
form          exp            # exp | power
species       2
Pb  1.553448  0.178537      # name W0 D
Ti  0.278416  0.098188
pairs         2
O Pb  2.072416 6.0 0.495635  # name1 name2 r0 C b
O Ti  1.760536 5.2 0.377942
```

Constraints: `newton on` required (EAM-like reverse communication);
serial + MPI supported; no GPU/KOKKOS variant.
