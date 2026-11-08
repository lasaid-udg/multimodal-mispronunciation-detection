# docs/METHODOLOGY_SCOPE.md
# Methodology Scope — What This Repository Reproduces

This document explicitly states what is and is not reproducible from this
repository, to avoid misleading claims to reviewers or readers.

---

## What IS reproducible

Starting from the **derived tabular feature dataset** (`data/derived/public_dataset.csv`
or `data/raw/Nuevo_final_df.csv`), this repository reproduces:

### ML analysis pipeline
- Loading and validating the dataset (SHA-256 verified)
- Sentinel-based channel exclusion (`888888` = unusable EMG for that participant)
- Construction of all 20 experimental scenarios (4 channels × 5 phoneme contrasts)
- Two modalities per scenario: Acoustic-only (`A`) and Acoustic+EMG (`A_CH{1–4}`)
- Leakage-safe nested cross-validation:
  - 5 outer folds × 3 inner folds × 10 repeats
  - `StratifiedGroupKFold` with `id_subject` grouping
  - No participant shared between train and test (outer or inner)
- 7 classifiers with their exact hyperparameter grids (Table 2 of manuscript):
  SVM, RF, DT, KNN, GNB, MLP, LR
- Pipeline per fold: `SimpleImputer(median)` → `VarianceThreshold(0)` → `StandardScaler`
  (scaler applied only for SVM, KNN, MLP, LR)
- `GridSearchCV` scoring: `balanced_accuracy`; `class_weight` tuned in inner CV
  for SVM, RF, DT, LR
- OOF predictions: each record appears exactly once per (repeat, model, modality)
- Metrics: balanced accuracy, sensitivity (class 1), specificity (class 0), MCC, ROC-AUC
- OOF metrics computed on the **full concatenated OOF set** per repeat (not fold averages)

### Statistical analysis
- Participant-cluster bootstrap: 2,000 resamples → 95% CI
- Paired participant-level permutation test: 2,000 permutations → p-value
- Benjamini-Hochberg FDR correction:
  - General family: 28 comparisons (7 models × 4 channels)
  - Phoneme-channel family: 112 comparisons (7 × 4 channels × 4 phonemes)

### Infrastructure
- Checkpointed, reanudable runs (SHA-256 + config validation)
- Full environment manifest per run (Python, library versions, git commit, dataset hash)

---

## What is NOT reproducible from this repository

### Signal acquisition and processing
- EMG surface electrode recording — hardware dependent
- Audio recording
- EMG artifact removal and segmentation — proprietary MATLAB toolbox
- Audio normalization and segmentation

### Feature extraction
The acoustic and EMG features in the dataset were extracted using a combination
of **MATLAB** and **Python** scripts that are **not included** in this repository.

Acoustic families extracted:
- Pitch, energy, formants, spectral features, MFCCs, shimmer, jitter, HNR, zero-crossing rate

EMG families extracted (per channel):
- Time-domain: Willison amplitude, zero crossings, mean absolute value, mean EMG
- Frequency-domain: median/mean frequency, spectral moments
- STFT: peak frequency, energy, frequency variance, centroid per point
- MFCC: 13 coefficients
- LPC: 16 coefficients

The extracted feature dataset is provided as-is. The `888888` sentinel values,
NaN values, and ±Inf values in LPC columns are characteristics of the original
extraction and are preserved exactly.

### Exploratory analyses
The following analyses are intentionally **separated from the core pipeline**
and are not required to reproduce the primary ML results:

- **Descriptive PCA** — visualisation only, not used for model fitting or inference
- **Permutation feature importance** — post-hoc descriptive analysis using
  already-fitted outer-fold models; requires `--save-models` during the run

These are available in `src/md_emg_extras/exploratory.py` and documented in
`docs/REPRODUCIBILITY.md`.

---

## Known limitations

| Limitation | Impact | Documented |
|------------|--------|-----------|
| Feature extraction not included | Cannot reproduce from raw signals | ✅ Here + README |
| Full reproduction run not yet completed | Tables S-5 to S-19 not numerically verified | ✅ README, REPRODUCIBILITY.md |
| MLP/LR may differ across OS/BLAS | Minor numerical differences possible | ✅ README |
| `id_word → english_word` mapping unverified | Documentation only; does not affect ML | ✅ phoneme_mapping.csv |
| No pre-trained models distributed | Must re-fit (≈ 50–100 CPU-h) | ✅ README |
| IRB/license pending | Cannot release raw recordings or finalize data license | ✅ DATA_RELEASE_REVIEW.md |

---

## What constitutes "reproducing the paper"

A full reproduction of the paper's ML analysis requires:

1. ✅ The derived tabular dataset (`public_dataset.csv` or `Nuevo_final_df.csv`)
2. ✅ This repository installed and smoke-tested
3. 🔴 Authorization and execution of the full run (~50–100 CPU-h):
   ```
   md-emg reproduce --channel {ch1,ch2,ch3,ch4} --phoneme {all,t,theta,d,eth} --n-jobs 8
   ```
4. 🔴 Running `md-emg analyze` on each of the 20 completed run directories
5. 🔴 Running `md-emg fdr` across all 20 analyzed runs
6. 🔴 Comparing `metrics_oof_summary.csv` and `analysis/paired_deltas.csv` against
   manuscript Tables S-5 to S-19 using `compare_runs.py` or a dedicated comparison script

Steps 3–6 are **blocked pending PI authorization** of compute resources.
The repository is structured so that these steps can be executed sequentially
without any code changes.
