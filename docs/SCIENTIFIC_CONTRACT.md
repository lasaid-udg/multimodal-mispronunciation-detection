# Experimental Protocol

**Paper:** *A Multimodal Machine Learning Approach to English Mispronunciation Detection
Using Acoustic and Surface Electromyographic Signals*
**Package version:** `md-emg v0.3.5.1`

This document describes the exact ML protocol implemented in this repository.
All parameters listed here are frozen and match the published paper.

---

## 1. Dataset

| Property | Value |
|---|---|
| File | `data/derived/public_dataset.csv` |
| SHA-256 | `89EB58339E77F6C92EB949474F2338D520C3BF627B0F0853A43CD0F00EE1EA43` |
| Total records | 1,558 |
| Total columns | 483 |
| Participants | 39 |
| Words (`id_word`) | 40 (values 1–40) |
| Class 0 — correct pronunciation | 1,013 |
| Class 1 — mispronunciation (positive class) | 545 |

**Grouping variable:** `id_subject` — no participant is shared between
training and test partitions, nor between outer and inner cross-validation.

### Feature structure

| Group | Columns | Description |
|---|---|---|
| Identifiers | 2 | `id_subject`, `id_word` |
| Acoustic features | 84 | `duration` through `zero_crossing_rate` |
| EMG CH1 features | 99 | Cricothyroid muscle |
| EMG CH2 features | 99 | Anterior belly of the digastric muscle |
| EMG CH3 features | 99 | Masseter muscle |
| EMG CH4 features | 99 | Zygomaticus major muscle |
| Target | 1 | `Clase` |

**Note:** `aqpq5_shimmer` is a known typographic variant of `apq5_shimmer` — preserved as-is.

### Sentinel value and channel exclusion

`888888` marks an unusable EMG recording for a given channel. A record is excluded
from channel CHx analyses if **any** of its CHx columns equals `888888`.
Exclusion is per channel — a record excluded from CH1 may still be used in CH2, CH3, CH4.

`±Inf` values are replaced with `NaN` before imputation (inside each training fold).

### Valid participant counts per channel

| Channel | Clean participants | Clean records |
|---|---|---|
| CH1 | 16 | 640 |
| CH2 | 31 | 1,238 |
| CH3 | 19 | 760 |
| CH4 | 22 | 880 |

### Phoneme word mapping

| Phoneme | Word IDs |
|---|---|
| /t/ | 1,3,5,7,9,12,14,16,18,20 |
| /θ/ | 2,4,6,8,10,11,13,15,17,19 |
| /d/ | 21,23,25,27,29,32,34,36,38,40 |
| /ð/ | 22,24,26,28,30,31,33,35,37,39 |

Each phoneme set contains exactly 10 word IDs; the four sets are mutually disjoint
and cover all 40 words.

---

## 2. Experimental scenarios

- **5 phoneme evaluations:** `all`, `/t/`, `/d/`, `/θ/`, `/ð/`
- **4 EMG channels:** CH1, CH2, CH3, CH4
- **Total:** 20 scenarios
- **2 modalities per scenario:** Acoustic-only (`A`) and Acoustic+EMG (`A_CHx`)
- Within each scenario, `A` and `A_CHx` use the same records, participants, and folds.

---

## 3. ML protocol

### 3.1 Preprocessing pipeline (per fold, no leakage)

1. `SimpleImputer(strategy="median")` — fitted on training partition only
2. `VarianceThreshold(0.0)` — fitted on training partition only
3. `StandardScaler()` — applied **only** to SVM, KNN, MLP, LR; not to RF, DT, GNB

### 3.2 Hyperparameter optimization

- `GridSearchCV` with `scoring="balanced_accuracy"`
- 3 inner folds (`StratifiedGroupKFold`, `shuffle=True`, `random_state = repeat_seed + outer_fold`)
- `class_weight` tuned (`None` vs `"balanced"`) for SVM, RF, DT, LR
- KNN: `weights = ["uniform", "distance"]` tuned; no class_weight
- GNB, MLP: no class_weight

### 3.3 Outer cross-validation

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat_seed)`
- **10 repetitions** — seed for repeat `r`: `2026 + (r − 1) × 10000`
- OOF predictions assembled from all 5 folds per (repeat, modality, model)
- **Metrics computed on the full OOF set**, not by averaging fold-level metrics

### 3.4 Seven classifiers

| Model | Scaling | Grid size | class_weight tuned |
|---|---|---|---|
| SVM | Yes | 12 | Yes (None, balanced) |
| RF | No | 12 | Yes (None, balanced) |
| DT | No | 48 | Yes (None, balanced) |
| KNN | Yes | 12 | No |
| GNB | No | 3 | No |
| MLP | Yes | 36 | No |
| LR | Yes | 6 | Yes (None, balanced) |

**MLP fixed settings:** `max_iter=3000`, `early_stopping=True`,
`validation_fraction=0.1`, `n_iter_no_change=20`.

**LR fixed settings:** `max_iter=2000`, `solver="lbfgs"`.

All grids are defined in [`configs/paper.yaml`](../configs/paper.yaml).

### 3.5 Metrics (computed on full OOF per repeat)

- Balanced accuracy (primary)
- Sensitivity (recall for class 1)
- Specificity (recall for class 0)
- MCC (Matthews correlation coefficient)
- ROC-AUC
- Average precision

---

## 4. Statistical analysis

### 4.1 Bootstrap CIs

- **Participant-level cluster bootstrap**, 2,000 replicates
- 95% percentile confidence intervals
- Metrics averaged across 10 repeats

### 4.2 Paired permutation test

- 2,000 permutations; paired A vs A+CHx within each model
- 50/50 subject-level swap — all records of a participant swap together
- Permutation p-value: `(count(|δ_perm| ≥ |δ_obs|) + 1) / (2001)`

### 4.3 FDR correction (Benjamini–Hochberg)

Applied **separately** to two families of comparisons:

| Family | Comparisons | Formula |
|---|---|---|
| General | 28 | 4 channels × 7 models |
| Phoneme-channel | 112 | 4 phonemes × 4 channels × 7 models |

α = 0.05. Input: permutation p-values for balanced accuracy (A+EMG − A).

---

## 5. Reproducibility

- Python 3.11; all dependency versions pinned in [`environment.yml`](../environment.yml)
- Each run records: dataset SHA-256, git commit, all library versions, all seeds,
  scenario definition, OOF predictions, metrics, and selected hyperparameters
- Runs are resumable; SHA-256 and configuration are verified before reusing checkpoints
