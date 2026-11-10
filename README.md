# md-emg: Mispronunciation Detection with Acoustic and EMG Features

Code and dataset for the paper:

> **A Multimodal Machine Learning Approach to English Mispronunciation Detection
> Using Acoustic and Surface Electromyographic Signals**
> — Arturo Guzmán-Vázquez, Israel Román-Godínez, Sulema Torres-Ramos
> — Universidad de Guadalajara, CUCEI

[![CI](https://github.com/lasaid-udg/multimodal-mispronunciation-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/lasaid-udg/multimodal-mispronunciation-detection/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE)

---

## What is this?

This repository contains:

- A **tabular feature dataset** of 1,558 English word pronunciations by 39 Spanish-speaking
  participants, with acoustic and surface EMG features extracted from synchronized recordings.
- A **machine learning pipeline** that trains and evaluates 7 classifiers to detect
  mispronunciations, using nested cross-validation with participant-level grouping.
- **Statistical analysis** tools: bootstrap confidence intervals, permutation tests,
  and FDR-corrected comparisons across 20 experimental scenarios.

The study asks whether adding EMG signals from speech-related muscles improves automatic
mispronunciation detection beyond acoustic features alone.

---

## Installation

```bash
# 1. Create the environment (Python 3.11, all dependencies pinned)
conda env create -f environment.yml
conda activate md-emg

# 2. Install the package
pip install -e .
```

---

## Quick start

```bash
# Verify the dataset (< 5 seconds)
md-emg validate-data

# Check that the installation works (< 2 minutes)
md-emg smoke

# Run one scenario — e.g., all phonemes, CH2 channel (31 participants, ~6–10 h)
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4

# Resume if interrupted
md-emg reproduce --channel ch2 --phoneme all --n-jobs 4 --resume runs/<run_dir>

# Compute bootstrap CIs and permutation p-values
md-emg analyze runs/<run_dir>
```

---

## Dataset

The dataset is included at `data/derived/public_dataset.csv` (10 MB).

| Property | Value |
|---|---|
| Rows | 1,558 — one per participant × word recording |
| Columns | 483 — participant ID, word ID, class label, 84 acoustic features, 396 EMG features |
| Participants | 39 |
| Words | 40 (10 per phoneme contrast: /t/, /θ/, /d/, /ð/) |
| Correct pronunciations | 1,013 |
| Mispronunciations | 545 |
| SHA-256 | `89EB58339E77F6C92EB949474F2338D520C3BF627B0F0853A43CD0F00EE1EA43` |

The dataset was derived from synchronized audio and surface EMG recordings.
Raw signals are not included — contact the corresponding author to request access.

Column descriptions: [`data/README.md`](data/README.md) · [`data/metadata/data_dictionary.md`](data/metadata/data_dictionary.md)

---

## Reproducing the paper

The paper reports results for 20 scenarios: 4 EMG channels × 5 phoneme groups
(all words, /t/, /d/, /θ/, /ð/). Each scenario compares two feature sets:
acoustic only (`A`) vs. acoustic + EMG (`A_CHx`).

```bash
# Linux / macOS — run all 20 scenarios
for channel in ch1 ch2 ch3 ch4; do
  for phoneme in all t d theta eth; do
    md-emg reproduce --channel $channel --phoneme $phoneme --n-jobs 4
  done
done

# Apply FDR correction across all results
md-emg fdr runs/
```

```powershell
# Windows
scripts\run_all_20_repeated.bat
```

**Total compute time: ~50–100 CPU-hours** depending on hardware.
Each run is checkpointed and can be resumed with `--resume`.

---

## What the pipeline does

- **Preprocessing** (inside each fold, no leakage): median imputation → variance filter → optional standardization
- **7 classifiers**: SVM, Random Forest, Decision Tree, k-NN, Gaussian Naïve Bayes, MLP, Logistic Regression
- **Nested CV**: 5 outer folds × 3 inner folds × 10 repeats, `StratifiedGroupKFold` by participant
- **Metrics**: balanced accuracy, sensitivity, specificity, MCC, ROC-AUC
- **Statistics**: 2,000-resample participant-cluster bootstrap → 95% CIs; 2,000-permutation paired test; BH-FDR on 28 + 112 comparisons

Full protocol: [`docs/SCIENTIFIC_CONTRACT.md`](docs/SCIENTIFIC_CONTRACT.md)

---

## Repository structure

```
md-emg/
├── src/md_emg/          # Pipeline: data loading, models, CV, metrics, analysis, FDR
├── configs/
│   ├── paper.yaml       # Exact configuration used in the paper
│   └── smoke.yaml       # Quick end-to-end check after installation
├── data/
│   ├── derived/         # public_dataset.csv (included)
│   └── metadata/        # Column dictionary, schema, statistics
├── tests/               # 48 automated tests
├── scripts/             # Batch scripts for running all 20 scenarios
├── docs/                # Protocol details and output file descriptions
├── environment.yml      # Pinned conda environment
└── pyproject.toml
```

---

## Output files

Each completed run directory contains:

| File | Contents |
|---|---|
| `manifest.json` | Python version, library versions, git commit, dataset SHA-256 |
| `metrics_oof.csv` | Out-of-fold metrics per repeat × modality × model |
| `metrics_oof_summary.csv` | Mean ± SD across 10 repeats |
| `predictions.parquet` | All OOF predictions |
| `best_params.csv` | Selected hyperparameters per fold |
| `analysis/` | Bootstrap CIs and permutation p-values |

---

## Tests

```bash
python -m pytest tests/ -v    # 48 tests — no dataset download required for most
md-emg validate-data          # 28 dataset integrity checks
```

---

## Limitations

- Feature extraction (MATLAB + Python scripts) is **not included**. The pipeline starts from the derived tabular dataset.
- MLP and LR may produce minor numerical differences across OS/BLAS versions even with the same seed.
- Raw recordings are not included. Contact the corresponding author to request access.

---

## License

**Code:** MIT — see [`LICENSE`](LICENSE)
**Dataset:** CC BY 4.0 — see [`LICENSE`](LICENSE)

---

## Citation

```bibtex
@article{guzman2026mdemg,
  title   = {A Multimodal Machine Learning Approach to English Mispronunciation
             Detection Using Acoustic and Surface Electromyographic Signals},
  author  = {Guzm{\'a}n-V{\'a}zquez, Arturo and
             Rom{\'a}n-God{\'i}nez, Israel and
             Torres-Ramos, Sulema},
  year    = {2026},
  note    = {Manuscript under review}
}
```
