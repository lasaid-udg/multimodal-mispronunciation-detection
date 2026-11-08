# Scientific Contract

**Project:** *A Multimodal Machine Learning Approach to English Mispronunciation Detection
Using Acoustic and Surface Electromyographic Signals*
**Package version:** `md-emg v0.3.5.1`
**Document status:** FROZEN — established in Phase 1 of repository preparation.

This document defines the invariants that **must not change** across all refactoring phases.
Any deviation from these invariants — whether discovered or introduced — must be documented as a
`proposed_scientific_deviation` and reviewed by the authors before being accepted or rejected.
Differences between this document and the code or manuscript are **incidents to investigate**,
not silent corrections.

---

## 1. Dataset

| Property | Value | Source |
|---|---|---|
| Input file | `data/raw/Nuevo_final_df.csv` | Manuscript §2.2.3 |
| SHA-256 | `98499BE472A8626F88B6330850ECBAB2218E9BE69B26F79D6A2502A789FFF2E6` | Measured Phase 1 |
| Total records | 1,558 | Manuscript §2.2.3 |
| Total columns | 491 | Measured Phase 0 |
| Participants | 39 | Manuscript §2.2.3 |
| Words (id_word) | 40 (values 1–40) | Measured Phase 0 |
| Class 0 (correct) | 1,013 | Manuscript §2.4 |
| Class 1 (error) | 545 | Manuscript §2.4 |
| Duplicate (subj, word) pairs | 0 | Measured Phase 0 |

**Class semantics:** `Clase = 1` → pronunciation error (positive class);
`Clase = 0` → correct pronunciation (negative class). The positive class is always 1.

**Grouping variable:** `id_subject` — participants must never be shared between training and test
partitions, nor between outer and inner cross-validation.

### 1.1 Feature structure (491 columns)

| Group | Count | Notes |
|---|---|---|
| Identifiers | 2 | `id_subject`, `id_word` |
| Acoustic features | 84 | `duration` through `zero_crossing_rate` (positional range in CSV) |
| EMG CH1 features | 101 | 99 predictors + 2 auxiliary (`order_CH1_cricotiroideo`, `time_CH1_cricotiroideo`) |
| EMG CH2 features | 101 | 99 predictors + 2 auxiliary (`order_CH2_digastrico`, `time_CH2_digastrico`) |
| EMG CH3 features | 101 | 99 predictors + 2 auxiliary (`order_CH3_masetero`, `time_CH3_masetero`) |
| EMG CH4 features | 101 | 99 predictors + 2 auxiliary (`order_CH4_cigomatico`, `time_CH4_cigomatico`) |
| Target | 1 | `Clase` |
| **Total** | **491** | |

**Known column name errata:** `aqpq5_shimmer` (expected `apq5_shimmer`). This name is preserved
as-is in all public releases for comparability. An alias is documented in the data dictionary only.

### 1.2 Auxiliary columns (excluded from predictors)

`order_CH1_cricotiroideo`, `order_CH2_digastrico`, `order_CH3_masetero`, `order_CH4_cigomatico`,
`time_CH1_cricotiroideo`, `time_CH2_digastrico`, `time_CH3_masetero`, `time_CH4_cigomatico`

These 8 columns must **not** be used as predictors. Their role and potential effect on results
must be verified before any public CSV omits them.

### 1.3 Sentinel value and channel exclusion rule

`888888` marks an unusable EMG recording for a given channel. The exclusion rule:

- A record is excluded from channel CHx analyses if **any** of its CHx columns equals `888888`.
- Exclusion is **per channel** — a record excluded from CH1 may still be used in CH2, CH3, CH4.
- `888888` must **never** be treated as a real measurement or imputed generically.
- `±inf` values are replaced with `NaN` in predictor columns only, after channel filtering,
  before learned imputation inside the pipeline.

### 1.4 Valid participant counts per channel (after sentinel exclusion)

| Channel | Clean subjects | Clean records (all phonemes) |
|---|---|---|
| CH1 | 16 | 640 |
| CH2 | 31 | 1,238 |
| CH3 | 19 | 760 |
| CH4 | 22 | 880 |

### 1.5 Phoneme word mapping (`PHONEME_WORDS`)

```python
PHONEME_WORDS = {
    "all":   None,                              # all 40 words
    "t":     [1,3,5,7,9,12,14,16,18,20],       # /t/
    "theta": [2,4,6,8,10,11,13,15,17,19],      # /θ/
    "d":     [21,23,25,27,29,32,34,36,38,40],  # /d/
    "eth":   [22,24,26,28,30,31,33,35,37,39],  # /ð/
}
```

- Each phoneme set contains exactly 10 `id_word` values.
- The four sets partition `{1, ..., 40}` completely and are mutually disjoint.
- The mapping `id_word` → English word is documented in Supplementary Material (S-1/S-2)
  and must not be inferred from column ordering without explicit evidence.

---

## 2. Experimental scenarios

- **5 phoneme evaluations:** `all`, `/t/`, `/d/`, `/θ/`, `/ð/`
- **4 EMG channels:** CH1, CH2, CH3, CH4
- **Total:** 20 scenarios
- **2 modalities per scenario:** Acoustic-only (`A`) and Acoustic+EMG (`A_CHx`)
- Within each scenario, `A` and `A_CHx` use **exactly the same records, participants, and folds**.

---

## 3. Frozen ML protocol

### 3.1 Pipeline per fold

1. `SimpleImputer(strategy="median")` — fitted on training partition only
2. `VarianceThreshold(0.0)` — fitted on training partition only
3. `StandardScaler()` — applied **only** to SVM, KNN, MLP, LR; not to RF, DT, GNB

### 3.2 Hyperparameter optimization

- `GridSearchCV` with `scoring="balanced_accuracy"`
- 3 inner folds (`StratifiedGroupKFold`, `shuffle=True`, `random_state = repeat_seed + outer_fold`)
- `class_weight` tuned (`None` vs `"balanced"`) for SVM, RF, DT, LR
- KNN: `weights = ["uniform", "distance"]` tuned (no class_weight)
- GNB, MLP: no class_weight

### 3.3 Outer evaluation

- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat_seed)`
- **10 repetitions** — seed for repeat `r`: `2026 + (r - 1) * 10000`
- For each (repeat, modality, model): OOF predictions assembled from all 5 folds
- **Metrics computed on the full OOF set**, not by averaging fold-level metrics

### 3.4 Seven classifiers and their grids (from `configs/paper.yaml`)

| Model | Scale | Grid size | class_weight |
|---|---|---|---|
| SVM | Yes | 12 | tuned (None, balanced) |
| RF | No | 12 | tuned (None, balanced) |
| DT | No | 48 | tuned (None, balanced) |
| KNN | Yes | 12 | — |
| GNB | No | 3 | — |
| MLP | Yes | 36 | — |
| LR | Yes | 6 | tuned (None, balanced) |

**MLP fixed settings:** `max_iter=3000`, `early_stopping=True`, `validation_fraction=0.1`,
`n_iter_no_change=20`. Early stopping uses only the training partition; it is an optimizer
stopping mechanism, not an evaluation split.

**LR fixed settings:** `max_iter=2000`, `solver="lbfgs"`.

### 3.5 Metrics (computed on full OOF per repeat)

- Balanced accuracy (primary)
- Sensitivity (recall for class 1, pronunciation errors)
- Specificity (recall for class 0, correct pronunciations)
- MCC (Matthews correlation coefficient)
- ROC-AUC (continuous score: `decision_function` for SVM; `predict_proba[:,1]` for others)
- Average precision

Undefined metrics (e.g., single-class fold) are set to `NaN` and documented.

---

## 4. Statistical analysis

### 4.1 Bootstrap

- **Participant-level cluster bootstrap**, 2,000 replicates
- Multinomial multiplicities (mathematically equivalent to sampling participants with replacement)
- Metrics computed within each OOF repeat, then **averaged across repeats**
- 95% percentile confidence intervals
- Seed: 2026 (independent from experiment seed; passed to `md-emg analyze --seed`)

### 4.2 Paired permutation test

- 2,000 permutations; paired A vs A+CHx within each model
- 50/50 subject-level swap — all records of a given participant swap together in every permutation
- Applied across all 10 repeats simultaneously (dependence preserved)
- Permutation p-value: `(count(|δ_perm| ≥ |δ_obs|) + 1) / (n_permutations + 1)`

### 4.3 FDR correction

- Benjamini-Hochberg procedure applied **separately** to two families:
  - **General family:** 28 comparisons (4 channels × 7 models)
  - **Phoneme-channel family:** 112 comparisons (4 phonemes × 4 channels × 7 models)
- α = 0.05
- Input: permutation p-values for balanced accuracy differences (A+EMG − A)

---

## 5. Reproducibility constraints

- Python 3.11; dependency versions pinned in `environment.yml` and `pyproject.toml`
- No version updates without compatibility verification for numerically sensitive operations
- Each run records: dataset SHA-256, code commit, resolved configuration, dependency versions,
  all seeds, scenario definition, OOF predictions (record_id, repeat, fold, y_true, y_pred,
  score), metrics, and selected hyperparameters
- Resume: SHA-256 and scientific configuration verified before reusing checkpoints
- `--no-save-models` allowed; OOF predictions and metrics can be reproduced without model files

---

## 6. Scope of the public repository

**Included:**
- Derived tabular dataset (491-column CSV, pending privacy review)
- Code for all primary ML analyses and statistical comparisons
- `paper.yaml` with frozen protocol

**Explicitly excluded:**
- Raw audio/EMG signals
- MATLAB signal processing scripts
- PCA visualization code (`exploratory.py`)
- Permutation feature importance code (`exploratory.py`)
- Diagnostic scripts (`check_ch3_gnb.py`, `check_lpc_extremes.py`, `check_pca_outlier.py`)
- ROC/PR figure generation script (`make_roc_pr_figure.py`)

---

## 7. Pending authorizations (must be resolved before public release)

- [ ] Privacy review of participant identifiers in the CSV
- [ ] Authorization to release derived dataset
- [ ] Choice of code license (MIT / Apache 2.0 / other)
- [ ] Choice of data license (CC BY 4.0 / CC BY-NC 4.0 / other)
- [ ] Confirmation of ethical approval citation format

---

*Established: Phase 1 — Freeze and baseline*
*Last modified: Phase 1*
