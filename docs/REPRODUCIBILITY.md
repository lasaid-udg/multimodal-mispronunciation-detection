# Reproducibility Guide

This document provides the exact commands to reproduce all 20 experimental scenarios
from the manuscript, run statistical analyses, and apply FDR correction.

> **Important:** The full reproduction requires substantial compute time.
> See §4 (Resource Estimates) before starting.
> Never run the full reproduction automatically — it must be a deliberate, authorized action.

---

## Prerequisites

```powershell
# 1. Activate environment (conda)
conda activate md-emg

# 2. Install package (if not already installed)
pip install -e .

# 3. Verify environment
python -c "import sklearn; print(sklearn.__version__)"  # Expected: 1.5.x
python -c "from md_emg.data import sha256_file; print(sha256_file('data/raw/Nuevo_final_df.csv'))"
# Expected: 98499BE472A8626F88B6330850ECBAB2218E9BE69B26F79D6A2502A789FFF2E6

# 4. Verify dataset
md-emg audit -c configs/paper.yaml
```

---

## 1. Software verification (smoke test)

**NOT FOR SCIENTIFIC INFERENCE. Run this first to confirm the installation works.**

```powershell
# Smoke test: 3 models, 2 folds, 1 repeat, CH2 general (~20 seconds)
md-emg run -c configs/smoke_phase1.yaml --n-jobs 2
```

Expected: run completes in <2 minutes; `runs_smoke/smoke_phase1_*/metrics_oof.csv` generated.

---

## 2. Full reproduction — 20 scenarios × 10 repetitions

### 2a. Sequential (Windows batch, recommended for logging)

```powershell
# From project root — runs all 20 scenarios sequentially
# Arguments: REPEATS N_JOBS [OUTPUT_ROOT]
scripts\run_all_20_repeated.bat 10 4
# Output: runs_final_r10\general_chX_r10_... and runs_final_r10\ph_PHO_chX_r10_...
```

### 2b. Individual scenarios (for parallel execution or resuming a single run)

```powershell
# General assessment (4 scenarios)
md-emg run -c configs/paper.yaml --channel ch1 --phoneme all --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch2 --phoneme all --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch3 --phoneme all --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch4 --phoneme all --repeats 10 --n-jobs 4 --output-root runs_final_r10

# Phoneme-level assessment (16 scenarios: 4 phonemes × 4 channels)
md-emg run -c configs/paper.yaml --channel ch1 --phoneme t     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch1 --phoneme d     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch1 --phoneme theta --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch1 --phoneme eth   --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch2 --phoneme t     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch2 --phoneme d     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch2 --phoneme theta --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch2 --phoneme eth   --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch3 --phoneme t     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch3 --phoneme d     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch3 --phoneme theta --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch3 --phoneme eth   --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch4 --phoneme t     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch4 --phoneme d     --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch4 --phoneme theta --repeats 10 --n-jobs 4 --output-root runs_final_r10
md-emg run -c configs/paper.yaml --channel ch4 --phoneme eth   --repeats 10 --n-jobs 4 --output-root runs_final_r10
```

### 2c. Resuming an interrupted run

```powershell
# Resume a specific interrupted run directory
md-emg run -c configs/paper.yaml --channel ch2 --phoneme all --repeats 10 --n-jobs 4 ^
    --output-root runs_final_r10 ^
    --resume runs_final_r10\general_ch2_r10_YYYYMMDD_HHMMSS_...
```

Resuming checks:
- Dataset SHA-256 must match
- Scientific configuration must be identical (seed, folds, models, scenario, grids)
- Completed fold units are reused; only missing units are retrained

---

## 3. Statistical analysis (post-hoc, no retraining)

### 3a. Per-scenario analysis

```powershell
# Analyze one scenario (bootstrap + permutation test)
md-emg analyze runs_final_r10\general_ch2_r10_... ^
    --bootstrap 2000 --permutations 2000 --seed 2026 --n-jobs 4

# Batch analysis of all 20 scenarios
scripts\analyze_runs.bat runs_final_r10 2000 2000 4
```

### 3b. FDR correction across all 20 scenarios

```powershell
# Requires all 20 runs to have completed analysis/paired_deltas.csv
md-emg fdr runs_final_r10
# Output: runs_final_r10\FDR_balanced_accuracy_all_models.csv
# Contains: 28 general + 112 phoneme-level BH-FDR corrected q-values
```

### 3c. Apply FDR via batch script

```powershell
scripts\apply_fdr.bat runs_final_r10
```

---

## 4. Resource estimates

> **Note:** These are approximate estimates based on the pipeline structure.
> Actual times depend on CPU speed, number of cores, and platform.
> Do NOT start the full reproduction without verifying available resources.

| Scenario type | Records | Models | Grid candidates | Outer fits | Est. time (4 jobs) |
|---|---|---|---|---|---|
| General CH2 (largest) | 1,238 | 7 | 129 total | 5×10=50 | ~6–10 h |
| General CH1 (smallest) | 640 | 7 | 129 total | 50 | ~3–5 h |
| Phoneme/ch (varies) | 100–600 | 7 | 129 total | 50 | ~1–8 h |
| **All 20 scenarios** | — | — | — | 1,000 | **~50–100 h total** |

**Storage estimate:** ~200–500 MB per scenario (with models saved), ~4–10 GB for all 20.
Use `--no-save-models` to skip model files (~10x storage reduction); OOF predictions are sufficient
for metrics, analysis, and FDR.

---

## 5. Verification against manuscript tables

After completing the full reproduction, compare results against the manuscript supplementary tables
(S-5 to S-19) using `compare_runs.py`:

```powershell
# Step 1: Prepare reference CSV (see template below)
# Step 2: Compare
python compare_runs.py runs_final_r10\general_ch2_r10_... ^
    --reference reference_results.csv --decimal-digits 3
```

**Reference CSV template** (`reference_results.csv`):
```csv
scenario,channel,phoneme,model,modality,metric,reported_value,source_table
general,ch2,all,svm,A,balanced_accuracy,0.565,Table_S5
general,ch2,all,svm,A_CH2,balanced_accuracy,0.545,Table_S5
...
```

> The manuscript reports values rounded to 3 decimal places.
> The comparator applies a tolerance of ±0.0005 (half of the last reported digit).
> Differences within rounding tolerance are labeled WITHIN_ROUNDING; larger differences
> are labeled OUTSIDE_TOLERANCE and require investigation.

---

## 6. Artifacts to preserve

After the full reproduction, retain for trazability:

| Artifact | Location | Purpose |
|---|---|---|
| `manifest.json` | Each run dir | Code version, SHA-256, environment |
| `predictions.parquet` | Each run dir | OOF predictions per record/repeat/fold |
| `splits.parquet` | Each run dir | Train/test assignments |
| `metrics_oof.csv` | Each run dir | Per-repeat OOF metrics |
| `analysis/paired_deltas.csv` | Each run dir | Permutation test results per model |
| `analysis/bootstrap_ci.csv` | Each run dir | Bootstrap CIs |
| `FDR_balanced_accuracy_all_models.csv` | `runs_final_r10/` | Final BH-FDR table |

Models (`models/*.joblib`) are large (~50–200 MB per scenario) and can be regenerated.
They are required only for permutation feature importance (`md-emg explore`).

---

## 7. Regression verification after refactoring

After each refactoring phase, verify that outputs are unchanged using `compare_runs.py`:

```powershell
# Compare baseline smoke run with new smoke run
python compare_runs.py ^
    baseline\smoke_phase1 ^
    runs_smoke\smoke_phase1_NEW_... ^
    --tol 1e-9
```

All 6 checks must PASS. Any MISMATCH is a regression candidate that must be investigated
before the refactored code is accepted.
