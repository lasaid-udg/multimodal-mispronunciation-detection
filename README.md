# md-emg: Multimodal Mispronunciation Detection — Reproducible ML Pipeline

Companion code and dataset for the paper:

> **A Multimodal Machine Learning Approach to English Mispronunciation Detection
> Using Acoustic and Surface Electromyographic Signals**
>
> *Manuscript under review. Citation information will be updated upon acceptance.*

[![CI](https://github.com/placeholder/md-emg/actions/workflows/ci.yml/badge.svg)](https://github.com/placeholder/md-emg/actions/workflows/ci.yml)

---

## What this repository is for

This repository lets you **reproduce the machine-learning results** reported in the paper.
It contains:

- **A derived tabular feature dataset** (`data/derived/public_dataset.csv`) — 1,558 pronunciation
  recordings × 483 columns (acoustic + EMG features, participant ID, word ID, and class label).
- **The complete ML analysis pipeline** — nested cross-validation, bootstrap confidence intervals,
  paired permutation tests, and Benjamini–Hochberg FDR correction.
- **Automated integrity checks** for the dataset (28 tests) and 51 unit/integration tests for
  the pipeline.

> **What this repository does NOT contain:**
> Raw audio and EMG recordings (available on request), feature-extraction scripts (MATLAB),
> or pre-trained model weights (models are re-fitted during reproduction).

---

## Installation

```bash
# 1. Create the environment with the exact library versions used in the paper
conda env create -f environment.yml
conda activate md-emg

# 2. Install the package
pip install -e .
```

Requires **Python 3.11**. All pinned versions are in [`environment.yml`](environment.yml).

---

## Quick start

### Step 1 — Verify the dataset

```bash
md-emg validate-data
```

Expected output: `28 passed, 0 failed`. This checks the CSV checksum, row/column counts,
class balance, and channel partitions — no training required.

### Step 2 — Run a smoke test

```bash
md-emg smoke
```

Runs a minimal end-to-end pipeline check (3 models, 2 outer folds, 1 repeat) in under
a minute. Confirms the installation works. **Results are not comparable to the paper.**

### Step 3 — Reproduce a scenario from the paper

Each of the 20 scenarios in the paper combines one EMG channel (CH1–CH4) and one phoneme
group (all words, /t/, /d/, /θ/, /ð/):

```bash
# Example: all phonemes, CH2 — the largest channel subset (31 participants)
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4

# Resume an interrupted run
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4 --resume runs/<run_dir>
```

After it finishes, apply the statistical analysis:

```bash
md-emg analyze runs/<run_dir>
```

### Step 4 — Reproduce all 20 scenarios

```bash
# Linux / macOS
for channel in ch1 ch2 ch3 ch4; do
  for phoneme in all t d theta eth; do
    md-emg reproduce --channel $channel --phoneme $phoneme --n-jobs 4
  done
done

# Then run statistical analysis and FDR correction
md-emg fdr runs/
```

On Windows, use the batch script:

```powershell
scripts\run_all_20_repeated.bat
```

Full reproduction of all 20 scenarios takes approximately **50–100 CPU-hours** depending
on hardware. The run can be safely interrupted and resumed with `--resume`.

---

## Dataset

| Property | Value |
|----------|-------|
| File | `data/derived/public_dataset.csv` |
| Rows | 1,558 (one per participant × word recording) |
| Columns | 483 (2 identifiers + 1 label + 84 acoustic + 396 EMG) |
| SHA-256 | `89EB58339E77F6C92EB949474F2338D520C3BF627B0F0853A43CD0F00EE1EA43` |
| Participants | 39 |
| Words | 40 (10 per phoneme contrast) |
| Class 0 — correct pronunciation | 1,013 rows |
| Class 1 — mispronunciation | 545 rows |
| Channel subsets | CH1: 640 rows (16 participants), CH2: 1,238 (31), CH3: 760 (19), CH4: 880 (22) |

**Target column:** `Clase` — binary (0 = correct, 1 = mispronunciation).

**Sentinel value:** `888888` marks EMG data unavailable for a given participant–word pair.
The pipeline excludes affected rows automatically before fitting.

Full column descriptions: [`data/README.md`](data/README.md) ·
[`data/metadata/data_dictionary.md`](data/metadata/data_dictionary.md)

---

## Experimental protocol

### 20 scenarios

Each scenario compares two modalities on the same participant–channel subset:

| Modality | Features used |
|----------|--------------|
| `A` | Acoustic only (84 columns) |
| `A_CH{1–4}` | Acoustic + EMG for that channel (84 + 99 = 183 columns) |

### 7 classifiers

SVM · Random Forest · Decision Tree · k-NN · Gaussian Naïve Bayes · MLP · Logistic Regression

Preprocessing inside each pipeline fold: `SimpleImputer(median)` → `VarianceThreshold(0)`
→ `StandardScaler` (scaler applied only for SVM, KNN, MLP, LR; tree-based models and GNB
are left unscaled).

### Nested cross-validation

| Parameter | Value |
|-----------|-------|
| Outer folds | 5 |
| Inner folds | 3 |
| Repeats | 10 |
| Grouping | `id_subject` — no participant shared between train and test sets |
| Base seed | 2026 |
| Selection metric | Balanced accuracy |

OOF metrics are computed on the **full concatenated** out-of-fold set per repeat,
not as an average over folds.

### Statistical analysis

- **Bootstrap (×2,000):** participant-cluster resampling → 95 % CI per metric
- **Permutation test (×2,000):** paired participant-level permutations → p-value for A+EMG vs A
- **FDR correction:** Benjamini–Hochberg separately on two families:
  - General: 28 comparisons (7 models × 4 channels)
  - Phoneme-channel: 112 comparisons (7 models × 4 channels × 4 phonemes)

Full protocol details: [`docs/SCIENTIFIC_CONTRACT.md`](docs/SCIENTIFIC_CONTRACT.md)

---

## Output files

Each completed `md-emg reproduce` run directory contains:

| File | Description |
|------|-------------|
| `manifest.json` | Python version, library versions, git commit, dataset SHA-256 |
| `config.yaml` | Effective configuration for this run |
| `predictions.parquet` | OOF predictions for all folds, repeats, and models |
| `metrics_oof.csv` | OOF metrics per repeat × modality × model |
| `metrics_oof_summary.csv` | Mean ± SD across 10 repeats |
| `best_params.csv` | Best hyperparameters per outer fold |
| `analysis/` | Bootstrap CIs and permutation p-values (from `md-emg analyze`) |

Full schema: [`docs/OUTPUTS.md`](docs/OUTPUTS.md)

---

## Repository structure

```
md-emg/
├── src/md_emg/               # Core pipeline
│   ├── data.py               # Dataset loading and scenario builder
│   ├── models.py             # 7 classifiers with pipelines and grids
│   ├── experiment.py         # Nested CV, seeds, checkpoints, resume
│   ├── metrics.py            # Balanced accuracy, sensitivity, specificity, MCC, AUC
│   ├── analysis.py           # Bootstrap CIs and permutation tests
│   ├── multiplicity.py       # BH-FDR (28 + 112 comparisons)
│   └── cli.py                # Commands: reproduce / analyze / fdr / validate-data / smoke
├── configs/
│   ├── paper.yaml            # Frozen paper protocol (5×3×10, seed=2026, 7 models)
│   └── smoke_phase1.yaml     # Fast end-to-end check (not for scientific use)
├── data/
│   ├── derived/public_dataset.csv    # Derived tabular dataset (committed)
│   └── metadata/                     # Column dictionary, schema, statistics
├── tests/                    # 48 tests (no CSV required for most)
├── scripts/                  # Batch scripts for all 20 scenarios
├── docs/                     # Protocol details, output schemas, methodology scope
├── environment.yml           # Pinned conda environment
└── pyproject.toml            # Package and tool configuration
```

---

## Running the tests

```bash
# Full test suite (51 tests — no CSV required)
python -m pytest tests/ -v

# Lint
python -m ruff check src/ tests/

# Dataset integrity (requires the CSV)
md-emg validate-data
```

---

## Limitations

- **Feature extraction is not included.** Acoustic and EMG features were extracted
  using MATLAB and Python scripts outside this repository. This pipeline reproduces
  only the ML analysis step, starting from the derived tabular dataset.
- **MLP and LR may show minor numerical differences** across OS and BLAS versions,
  even with the same seed, due to floating-point non-associativity in parallel operations.
  Results should be within the reported standard deviations.
- **Raw recordings are not included.** The dataset contains derived features only.
  Contact the corresponding author to request access to raw signals.

---

## Data availability

The derived tabular dataset is distributed with this repository.
Raw biometric recordings (audio and surface EMG) are not included and require
a data sharing agreement. Contact the corresponding author.

---

## License

**Code:** to be added before public release.

**Dataset:** to be confirmed pending institutional review.

---

## Citation

```bibtex
@misc{md-emg-2026,
  title  = {A Multimodal Machine Learning Approach to English Mispronunciation
            Detection Using Acoustic and Surface Electromyographic Signals},
  author = {[Author names to be confirmed]},
  year   = {2026},
  note   = {Manuscript under review. Repository: https://github.com/placeholder/md-emg}
}
```

*DOI and full citation will be updated upon acceptance.*
