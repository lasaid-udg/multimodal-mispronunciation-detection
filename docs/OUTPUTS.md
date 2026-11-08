# docs/OUTPUTS.md
# Run Output Files — Schema and Contracts

This document describes every file produced by `md-emg run` / `md-emg reproduce`
inside a run directory, including their schema, invariants, and how they relate
to the reproducibility contract.

---

## Run directory structure

```
runs/<run_id>/
├── manifest.json               # Environment + dataset provenance
├── config.yaml                 # Effective (resolved) configuration
├── audit.json                  # Pre-run dataset audit snapshot
├── features.json               # Feature column names by modality
├── splits.parquet              # Train/test assignment per record × repeat × fold
├── predictions.parquet         # OOF predictions (all folds concatenated)
├── metrics_folds.csv           # Per-fold metrics (before OOF aggregation)
├── metrics_oof.csv             # OOF metrics per repeat × modality × model
├── metrics_oof_summary.csv     # mean ± SD across repeats per modality × model
├── best_params.csv             # Best hyperparameters per fold unit
├── fold_outputs/               # Per-unit checkpoint files (atomic)
│   ├── pred_r<R>_<model>_<modality>_f<F>.parquet
│   ├── metric_r<R>_<model>_<modality>_f<F>.json
│   └── params_r<R>_<model>_<modality>_f<F>.json
├── inner_cv/                   # GridSearchCV cv_results_ per unit
│   └── r<R>_<model>_<modality>_f<F>.parquet
├── models/                     # Fitted outer-fold estimators (optional)
│   └── r<R>_<model>_<modality>_f<F>.joblib
├── logs/
│   └── run.log                 # Full timestamped run log
└── analysis/                   # Produced by `md-emg analyze`
    ├── repeat_summary.csv
    ├── bootstrap_ci.csv
    ├── permutation_deltas.csv
    └── analysis_manifest.json
```

---

## File schemas

### `manifest.json`

Saved once at the start of a fresh run. Never overwritten during resume.

| Field | Type | Description |
|-------|------|-------------|
| `python` | str | Full Python version string |
| `platform` | str | OS and architecture |
| `hostname` | str | Machine hostname |
| `numpy` | str | numpy version |
| `pandas` | str | pandas version |
| `scipy` | str | scipy version |
| `scikit_learn` | str | scikit-learn version |
| `joblib` | str | joblib version |
| `threadpoolctl` | str | threadpoolctl version |
| `git_commit` | str | Short git commit hash (`"unknown"` if not in a repo) |
| `command` | str | Full CLI command that launched the run |
| `run_id` | str | Unique run identifier |
| `data_path` | str | Absolute path to dataset used |
| `data_sha256` | str | SHA-256 hex digest of the dataset (lowercase) |

**Invariant:** `data_sha256` is verified on resume — mismatches abort the run.

---

### `config.yaml`

The fully-resolved configuration after all CLI overrides are applied.
This is the single source of truth for the scientific protocol of this run.

**Invariant:** The scientific subset (`experiment`, `data`, `scenario`, `cv`,
`models`, `balance`, `selection`) is compared against the saved config on
resume — mismatches abort the run.

---

### `splits.parquet`

One row per (record, repeat, fold, role).

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | str | `s{id_subject}_w{id_word}` |
| `id_subject` | int | Participant ID |
| `repeat` | int | Outer CV repeat (1-indexed) |
| `fold` | int | Outer CV fold (1-indexed) |
| `role` | str | `"train"` or `"test"` |

**Invariants:**
- For every (repeat, fold): train and test subject sets are disjoint.
- Every record appears exactly once as `"test"` per repeat.

---

### `predictions.parquet`

Concatenated OOF predictions across all folds, repeats, modalities, and models.

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | str | `s{id_subject}_w{id_word}` |
| `id_subject` | int | Participant ID |
| `id_word` | int | Word ID |
| `repeat` | int | Outer CV repeat (1-indexed) |
| `fold` | int | Outer CV fold (1-indexed) |
| `modality` | str | `"A"` or `"A_CH{1-4}"` |
| `model` | str | One of: `svm`, `rf`, `dt`, `knn`, `gnb`, `mlp`, `lr` |
| `selected_class_weight` | str/None | Class weight chosen by inner CV |
| `y_true` | int | Ground truth label (0 or 1) |
| `y_pred` | int | Predicted label (0 or 1) |
| `score_error` | float | Continuous probability/score for class 1 (NaN if unavailable) |
| `score_type` | str | `"predict_proba"`, `"decision_function"`, or `"none"` |

**Invariants:**
- For every (repeat, modality, model): each `record_id` appears exactly once.
- `y_true` values must be 0 or 1.
- `score_error` is used for ROC-AUC; its sign follows scikit-learn's convention.

---

### `metrics_oof.csv`

OOF metrics computed from the **full concatenated OOF set** per (repeat, modality, model).
**Not** an average of per-fold metrics.

| Column | Description |
|--------|-------------|
| `run_id` | Run identifier |
| `repeat` | Outer CV repeat |
| `modality` | `A` or `A_CH{1-4}` |
| `model` | Classifier name |
| `balanced_accuracy` | Primary metric (manuscript Table 3 / S-5 to S-14) |
| `sensitivity` | Recall for class 1 (mispronunciation) |
| `specificity` | Recall for class 0 (correct) |
| `mcc` | Matthews Correlation Coefficient |
| `roc_auc` | Area under ROC curve |
| `f1_error` | F1 for class 1 |
| `accuracy` | Overall accuracy |
| `precision_error` | Precision for class 1 |
| `fpr` | False positive rate |
| `average_precision` | Average precision (PR-AUC) |
| `predicted_error_rate` | Fraction predicted as class 1 |

---

### `metrics_oof_summary.csv`

Mean ± SD across repeats for each (modality, model) combination.

Columns: `run_id`, `modality`, `model`, `n_repeats`, then for each metric:
`{metric}_mean` and `{metric}_sd`.

**Note:** SD uses `ddof=1`. If `n_repeats=1`, SD is NaN.

---

### `best_params.csv`

Best hyperparameters selected by inner CV for each outer fold unit.

| Column | Description |
|--------|-------------|
| `run_id`, `repeat`, `modality`, `model`, `fold` | Unit identifier |
| `best_inner_score` | Best inner CV balanced accuracy |
| `selected_class_weight` | Class weight chosen |
| `best_params` | JSON string of all best parameters |
| `n_iter`, `max_iter` | Iteration info (MLP/LR/SVM) |
| `reached_max_iter` | Convergence flag (True = did NOT converge) |
| `best_validation_score` | MLP early stopping best score |
| `convergence_warning_count` | Number of ConvergenceWarnings captured |

---

### `fold_outputs/pred_r<R>_<model>_<modality>_f<F>.parquet`

Per-unit checkpoint. Subset of `predictions.parquet` for one (repeat, model,
modality, fold). Written atomically before being acknowledged as complete.

**Resume invariant:** A unit is considered complete only when ALL three files
exist: `pred_*.parquet`, `metric_*.json`, `params_*.json`.

---

## Analysis outputs (`md-emg analyze`)

Produced in `<run_dir>/analysis/`.

| File | Description |
|------|-------------|
| `repeat_summary.csv` | OOF metrics per repeat (mirrors `metrics_oof.csv` with CI columns) |
| `bootstrap_ci.csv` | 95% bootstrap CIs, participant-clustered, 2000 resamples |
| `permutation_deltas.csv` | Paired A+EMG minus A deltas, permutation p-values, 2000 permutations |
| `analysis_manifest.json` | Bootstrap/permutation settings, seeds, data hash |

---

## FDR output (`md-emg fdr`)

| File | Description |
|------|-------------|
| `fdr_results.csv` (default) | BH-FDR adjusted p-values for 28 general + 112 phoneme BA comparisons |

---

## What is NOT saved by default

| Artifact | Reason | How to enable |
|----------|--------|---------------|
| Fitted `.joblib` models | Large; not needed to verify metrics | Remove `--no-save-models` |
| Inner CV results | Can be large; useful for diagnostics | `output.save_inner_cv: true` in config |

---

## Checkpoint / resume contract

A run directory is **safe to resume** when:
1. `config.yaml` exists and its scientific subset matches the new config.
2. `manifest.json` exists and its `data_sha256` matches the current dataset.
3. Each fold unit in `fold_outputs/` has all three checkpoint files.

A run directory is **NOT safe to resume** when:
- The dataset has changed (different SHA-256).
- The scientific config has changed (different CV params, models, scenario, seed).
- The checkpoint files are partial (only pred OR metric, not both).

The `--resume <run_dir>` flag to `md-emg run` / `md-emg reproduce` enforces
these checks automatically and raises a `RuntimeError` on any violation.
