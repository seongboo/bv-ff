# scripts/

Utility scripts that sit outside the main `src/` pipeline.

| File | Type | Purpose |
|------|------|---------|
| `gen_controls.py`   | CLI | Generate a `controls.toml` with sensible defaults. |
| `analysis.py`       | library | RMSE / parity / force-distribution plots. Imported by `src/main.py` (Step 6/7). |
| `ferroelectric.py`  | CLI + library | Ferroelectric validation gate: double-well scan, point-charge polarization, polar retention, NVT stability. `run_validation()` runs as Step 7/7 of every fit and persists `ferroelectric_validation.toml` + `double_well.png`; run standalone from a fit directory for a nonzero exit code on failure. |
| `gen_poscars.py`    | CLI | Generate the static DFT training-structure set (cubic EOS, tetragonal GS, double-well path, c/a scan) as POSCARs. |
| `vasprun2data.py`   | CLI | Convert vasprun.xml trajectories into the long-format AIMD parquet the readers ingest. |

---

## gen_controls.py

CLI that emits a valid `controls.toml` for the current dataset schema. Run it once when starting a new system or whenever you want to sweep settings.

### Quick start

```bash
# Single vasprun (default file name)
python scripts/gen_controls.py
# → writes controls.toml with dataset = "vasprun.xml"

# Single extxyz / vasprun explicitly
python scripts/gen_controls.py path/to/file.extxyz

# Multiple files, shared window  →  dataset = ["a", "b", ...]
python scripts/gen_controls.py data/300K.extxyz data/600K.extxyz \
    --frame-start 100 --stride 10

# Per-file window (table-array form) so each file can be edited independently
python scripts/gen_controls.py data/300K.extxyz data/900K.extxyz --per-file
```

### Behavior

- One positional path → `dataset = "..."`.
- Multiple positional paths → `dataset = [...]` (each inherits the top-level window).
- `--per-file` → `[[dataset]]` blocks at the bottom of the file; each block seeded with the top-level window, edit per-file as needed.
- Refuses to overwrite an existing output file; pass `-f`/`--force` to overwrite.
- Warns (does not error) when a dataset path doesn't exist yet, so you can scaffold a config before producing the data. Disable with `--no-check`.

### Options

| Group       | Flag                        | Default       |
|-------------|-----------------------------|---------------|
| output      | `-o`, `--output`            | `controls.toml` |
|             | `-f`, `--force`             | off           |
|             | `--per-file`                | off           |
|             | `--no-check`                | off           |
| window      | `--frame-start`             | `0`           |
|             | `--frame-end`               | `-1` (all)    |
|             | `--stride`                  | `1`           |
| run         | `--task` {energy,force,both}| `both`        |
|             | `--train-ratio`             | `0.8`         |
|             | `--split-mode` {random,block}| `random` (`block` = honest holdout) |
|             | `--output-dir`              | `./output`    |
|             | `--log-file`                | `bvff.log`    |
| potentials  | `--no-coulomb`              | on            |
|             | `--no-repulsive`            | on            |
|             | `--buckingham`              | off (replaces r⁻¹²; auto-disables repulsive) |
|             | `--no-bv`                   | on            |
|             | `--no-bvv`                  | on            |
|             | `--angle`                   | off           |
|             | `--bv-form` {exp,power}     | `exp`         |
| extensions  | `--no-ewald`                | on            |
| fitting     | `--w-E`, `--w-F`, `--w-S`   | `1.0`         |
|             | `--use-stress`              | off           |
|             | `--fit-charges`             | off (recommended) |
|             | `--lambda-reg`              | `0.0` (try ~1e-1 if a fit saturates bounds) |
|             | `--target-loss`             | `0.0` (off)   |
|             | `--patience`                | `0` (off)     |
|             | `--maxiter`                 | `1000`        |

Run `python scripts/gen_controls.py --help` for the full list.

### Examples

```bash
# Force-only fit, drop BVV, heavier force weight
python scripts/gen_controls.py vasprun.xml \
    --no-bvv --task force --w-F 2.0

# Direct Coulomb (no Ewald), angle term on
python scripts/gen_controls.py vasprun.xml \
    --no-ewald --angle

# Multi-temperature, per-file window
python scripts/gen_controls.py \
    data/300K.extxyz data/600K.extxyz data/900K.extxyz \
    --per-file
# → edit the [[dataset]] blocks in controls.toml to tune each file
```

---

## analysis.py

Not a standalone script — it is imported by `src/main.py` and runs as Step 6/7 of the pipeline. After fitting, it produces:

| File (in `output_dir/`) | Contents |
|-------------------------|----------|
| `parity_plot.png`       | Energy & force parity (BVFF vs AIMD), train and test panels. |
| `force_distribution.png`| Histogram of `|F|` for BVFF vs AIMD, train and test. |
| `rmse_summary.txt`      | Train/test RMSE for energy (eV/atom) and forces (eV/Å). |

Reusable functions if you want to call analysis from your own script:

```python
from scripts.analysis import (
    collect_predictions,   # bvff, frames → dict of e_*/f_* arrays
    compute_rmse,          # (pred, ref)  → float
    plot_parity,           # train_data, test_data, output_dir
    plot_force_distribution,
    run_analysis,          # the full pipeline used by main.py
)
```
