# Scope of Reproduction

This repository reproduces **the machine learning analysis** step of the paper,
starting from the provided tabular feature dataset.

---

## What this repository reproduces

### ML analysis pipeline

- Loading the dataset and applying sentinel-based channel exclusion (`888888`)
- Construction of all 20 experimental scenarios (4 EMG channels × 5 phoneme groups)
- Two modalities per scenario: Acoustic-only (`A`) and Acoustic+EMG (`A_CHx`)
- Nested cross-validation (5 outer × 3 inner × 10 repeats) with `StratifiedGroupKFold` —
  no participant is shared between training and test partitions
- 7 classifiers with the exact hyperparameter grids from the paper:
  SVM, RF, DT, KNN, GNB, MLP, LR
- Out-of-fold metrics: balanced accuracy, sensitivity, specificity, MCC, ROC-AUC

### Statistical analysis

- Participant-cluster bootstrap (×2,000) → 95% confidence intervals
- Paired participant-level permutation test (×2,000) → p-values
- Benjamini–Hochberg FDR correction over two families:
  - General: 28 comparisons (7 models × 4 channels)
  - Phoneme-channel: 112 comparisons (7 × 4 channels × 4 phonemes)

---

## What this repository does NOT include

| Item | Reason |
|------|--------|
| Raw audio and EMG recordings | Available on request from the corresponding author |
| Signal preprocessing scripts | Hardware-specific; depend on proprietary MATLAB toolbox |
| Feature extraction scripts | Implemented in MATLAB and Python; not distributed |
| Pre-trained model weights | Models are re-fitted during reproduction |

The derived tabular dataset (`data/derived/public_dataset.csv`) is sufficient to
reproduce all ML results and statistical tables reported in the paper.

---

## Known numerical limitations

- **MLP and LR** may show minor differences across operating systems and BLAS
  implementations due to floating-point non-associativity in parallel operations,
  even with the same random seed. Results should fall within the reported standard deviations.
- **Full reproduction requires ~50–100 CPU-hours.** Each run is checkpointed and
  can be safely interrupted and resumed.
