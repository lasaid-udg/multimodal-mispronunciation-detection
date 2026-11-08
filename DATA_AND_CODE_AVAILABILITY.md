# DATA_AND_CODE_AVAILABILITY.md
# Data and Code Availability Statement

*This document provides a ready-to-adapt statement for the manuscript's
Data and Code Availability section. Authors should review and approve
all claims before submission.*

---

## Suggested statement for the manuscript

> The code for the primary machine learning analyses (nested cross-validation,
> bootstrap confidence intervals, paired permutation tests, and Benjamini-Hochberg
> FDR correction) is publicly available at
> [https://github.com/placeholder/md-emg](https://github.com/placeholder/md-emg)
> (version 0.3.5.1, DOI: *to be assigned*).
>
> A derived tabular feature dataset (1,558 pronunciation recordings × 483 acoustic
> and EMG columns) is distributed with the repository.
> The raw biometric recordings (audio and surface EMG signals) are not publicly
> available due to [IRB restrictions / data-sharing policy — specify].
> Requests for access to the raw data should be directed to the corresponding author.
>
> Feature extraction was performed in MATLAB and Python. Those scripts are not
> included in the public repository because they depend on proprietary signal
> processing tools. The derived tabular dataset is sufficient to fully reproduce
> the ML analyses reported in this paper.

---

## What is distributed

| Artifact | Location | Distributed? |
|----------|----------|-------------|
| Derived feature dataset (483 cols) | `data/derived/public_dataset.csv` | ✅ Yes (pending license) |
| Data dictionary (491 columns) | `data/metadata/data_dictionary.{csv,md}` | ✅ Yes |
| ML pipeline code | `src/md_emg/` | ✅ Yes |
| Frozen paper configuration | `configs/paper.yaml` | ✅ Yes |
| Tests and CI | `tests/`, `.github/` | ✅ Yes |
| Raw audio recordings (.wav) | — | ❌ No |
| Raw EMG recordings | — | ❌ No |
| Original 491-col CSV | `data/raw/Nuevo_final_df.csv` | ❌ Requires data agreement |
| Feature extraction scripts (MATLAB) | — | ❌ Not included |
| Pre-trained model weights | — | ❌ Not persisted |
| Exploratory analyses (PCA, permutation importance) | `src/md_emg_extras/` | ⚠️ Available but not primary |

---

## What is NOT reproducible from this repository

The following steps from the full study pipeline are **not** reproducible from
this repository alone:

1. **Signal acquisition** — EMG surface electrodes and audio recording hardware.
2. **Preprocessing** — EMG artifact removal, segmentation, and audio normalization.
3. **Feature extraction** — Acoustic features (openSMILE / Praat-style),
   EMG features (time-domain, frequency-domain, STFT, MFCC, LPC).
   These were computed in MATLAB and Python using tools not included here.

The repository reproduces **from the derived tabular features** to the final
statistical tables, which is the primary contribution of the ML analysis.

---

## Verification status

| Claim | Status |
|-------|--------|
| Dataset integrity (28 tests) | ✅ PASS |
| 483-col ≡ 491-col equivalence for all 20 ML scenarios | ✅ PASS |
| Pipeline runs end-to-end (smoke test) | ✅ PASS |
| 51 unit/integration tests | ✅ PASS |
| Numerical results match manuscript Tables S-5 to S-19 | 🔴 BLOCKED — full run not yet executed |
| IRB approval for distributing derived features | 🔴 BLOCKED — pending |
| Data license assigned | 🔴 BLOCKED — pending |

---

## Checklist before submitting this statement

- [ ] Replace `[https://github.com/placeholder/md-emg]` with the real GitHub URL
- [ ] Replace `DOI: to be assigned` with the Zenodo/OSF DOI after archiving
- [ ] Confirm IRB protocol number and data-sharing policy
- [ ] Complete the full reproduction run and verify Tables S-5 to S-19
- [ ] Get PI sign-off on the exact wording
- [ ] Confirm author list and corresponding author contact
