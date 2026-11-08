# Reproducibility Guide

This document provides step-by-step instructions to reproduce all 20 experimental
scenarios reported in the paper, run the statistical analysis, and apply FDR correction.

---

## Prerequisites

```bash
# 1. Create and activate the environment
conda env create -f environment.yml
conda activate md-emg

# 2. Install the package
pip install -e .

# 3. Verify the installation
md-emg smoke
# Expected: run completes in < 2 minutes with no errors

# 4. Verify the dataset
md-emg validate-data
# Expected: 28 passed, 0 failed
```

> The full reproduction requires substantial compute time (~50–100 CPU-hours total).
> See [Resource estimates](#resource-estimates) before starting.
> Each run can be safely interrupted and resumed with `--resume`.

---

## Running the 20 scenarios

### All 20 scenarios at once (Windows)

```powershell
scripts\run_all_20_repeated.bat
```

### All 20 scenarios (Linux / macOS)

```bash
for channel in ch1 ch2 ch3 ch4; do
  for phoneme in all t d theta eth; do
    md-emg reproduce --channel $channel --phoneme $phoneme --n-jobs 4
  done
done
```

### Individual scenarios

```bash
# General assessment (one channel at a time)
md-emg reproduce --channel ch1 --phoneme all --n-jobs 4
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4
md-emg reproduce --channel ch3 --phoneme all --n-jobs 4
md-emg reproduce --channel ch4 --phoneme all --n-jobs 4

# Phoneme-level assessment (16 scenarios: 4 phonemes × 4 channels)
md-emg reproduce --channel ch1 --phoneme t     --n-jobs 4
md-emg reproduce --channel ch1 --phoneme d     --n-jobs 4
md-emg reproduce --channel ch1 --phoneme theta --n-jobs 4
md-emg reproduce --channel ch1 --phoneme eth   --n-jobs 4
# ... repeat for ch2, ch3, ch4
```

### Resuming an interrupted run

```bash
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4 \
    --resume runs/<run_directory_name>
```

---

## Statistical analysis

After each scenario completes, run the bootstrap and permutation analysis:

```bash
# Analyze one run directory
md-emg analyze runs/<run_directory_name>

# Analyze all runs in batch (Windows)
scripts\analyze_runs.bat runs 2000 2000 4
```

---

## FDR correction

After all 20 scenarios have been analyzed:

```bash
md-emg fdr runs/
```

Output: `runs/FDR_balanced_accuracy_all_models.csv`
Contains BH-FDR corrected q-values for all 28 general + 112 phoneme-level comparisons.

```powershell
# Windows batch
scripts\apply_fdr.bat runs
```

---

## Resource estimates

| Scenario | Records | Approx. time (4 cores) |
|----------|---------|----------------------|
| General CH1 | 640 | ~3–5 h |
| General CH2 | 1,238 | ~6–10 h |
| General CH3 | 760 | ~3–6 h |
| General CH4 | 880 | ~4–7 h |
| Phoneme-level | 100–600 | ~1–8 h each |
| **All 20 scenarios** | — | **~50–100 h total** |

**Storage:** ~20–50 MB per scenario without saved models. Use `--no-save-models`
(already the default in `configs/paper.yaml`) to save space; the OOF predictions
and metrics are sufficient to reproduce all tables.

---

## Output files

Each completed run directory contains:

| File | Description |
|------|-------------|
| `manifest.json` | Environment: Python version, library versions, git commit, dataset SHA-256 |
| `config.yaml` | Full configuration used for this run |
| `predictions.parquet` | OOF predictions (all folds, repeats, models) |
| `metrics_oof.csv` | OOF metrics per repeat × modality × model |
| `metrics_oof_summary.csv` | Mean ± SD across 10 repeats — corresponds to Tables S-5 to S-14 |
| `best_params.csv` | Best hyperparameters selected per outer fold |
| `analysis/bootstrap_ci.csv` | 95% confidence intervals per model and modality |
| `analysis/paired_deltas.csv` | Permutation test results (A+EMG vs A) per model |

Full schema: [`docs/OUTPUTS.md`](OUTPUTS.md)
