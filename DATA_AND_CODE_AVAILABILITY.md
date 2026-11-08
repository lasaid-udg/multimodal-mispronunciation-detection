# Data and Code Availability

The code for the machine learning analyses reported in this paper — including nested
cross-validation, bootstrap confidence intervals, paired permutation tests, and
Benjamini–Hochberg FDR correction — is publicly available at:

> [https://github.com/placeholder/md-emg](https://github.com/placeholder/md-emg)
> (version 0.3.5.1)

A derived tabular feature dataset (1,558 pronunciation recordings × 483 acoustic and EMG
columns) is distributed with the repository under the terms described in the LICENSE file.

The raw biometric recordings (audio and surface electromyographic signals) are not publicly
distributed. Requests for access to the raw data should be directed to the corresponding author.

Feature extraction was performed using MATLAB and Python scripts that are not included in
the public repository. The derived tabular dataset is sufficient to reproduce all ML analyses
reported in the paper.

---

## What this repository contains

| Artifact | Description |
|----------|-------------|
| `data/derived/public_dataset.csv` | Derived feature dataset (1,558 × 483) |
| `data/metadata/` | Column dictionary, schema, and summary statistics |
| `src/md_emg/` | Complete ML analysis pipeline |
| `configs/paper.yaml` | Exact configuration used in the paper |
| `tests/` | 51 automated tests |
| `.github/workflows/ci.yml` | Continuous integration workflow |

## What is not included

| Item | Reason |
|------|--------|
| Raw audio recordings | Not distributed; available on request |
| Raw EMG recordings | Not distributed; available on request |
| Feature extraction scripts | Depend on proprietary tools |
| Pre-trained model weights | Re-fitted during reproduction |
