# data/

This directory contains the dataset used in the paper
*"A Multimodal Machine Learning Approach to English Mispronunciation Detection
Using Acoustic and Surface Electromyographic Signals"*.

---

## Directory structure

```
data/
├── raw/
│   └── Nuevo_final_df.csv          # Original dataset (491 cols) — NOT distributed publicly
├── derived/
│   ├── public_dataset.csv          # Public version (483 cols) — 8 auxiliary cols removed
│   └── public_dataset_manifest.json  # Provenance manifest with SHA-256 checksums
└── metadata/
    ├── data_dictionary.csv         # Full column descriptions (491 entries)
    ├── data_dictionary.md          # Human-readable version of the dictionary
    ├── schema.json                 # Dataset schema (types, ranges, constraints)
    ├── dataset_statistics.json     # Verified summary statistics (matched to manuscript)
    ├── dataset_audit.json          # Per-column audit report (nulls, sentinels, inf)
    └── phoneme_mapping.csv         # word id → phoneme category mapping (40 rows)
```

---

## Datasets

### `raw/Nuevo_final_df.csv` — Original dataset (NOT DISTRIBUTED)

| Property | Value |
|----------|-------|
| Rows | 1,558 |
| Columns | 491 |
| SHA-256 | `98499BE472A8626F88B6330850ECBAB2218E9BE69B26F79D6A2502A789FFF2E6` |
| Participants | 39 |
| Words | 40 |
| Class 0 (correct) | 1,013 |
| Class 1 (mispronounced) | 545 |

> **Note:** This file is excluded from version control (see `.gitignore`).
> It must be placed here before running the pipeline. Contact the corresponding
> author to obtain the data under a data sharing agreement.

The 8 extra columns relative to the public version are auxiliary ordering/timing
columns (`order_CH1`, `time_CH1`, `order_CH2`, `time_CH2`, `order_CH3`,
`time_CH3`, `order_CH4`, `time_CH4`). They are **not used** by the ML pipeline.

---

### `derived/public_dataset.csv` — Public distribution version

| Property | Value |
|----------|-------|
| Rows | 1,558 |
| Columns | 483 |
| SHA-256 | `89EB58339E77F6C92EB949474F2338D520C3BF627B0F0853A43CD0F00EE1EA43` |
| Derived from | `raw/Nuevo_final_df.csv` (drop 8 auxiliary columns) |

This is a **lossless** derivation: the 8 dropped columns carry no information
used by any ML scenario. Equivalence between the two versions has been
verified programmatically for all 20 experimental scenarios
(see `scripts/prepare_public_dataset.py`).

To reproduce the derivation deterministically:

```bash
python scripts/prepare_public_dataset.py
```

To validate the file integrity:

```bash
# Public version (483 cols)
python scripts/validate_dataset.py

# Original (491 cols, requires raw file)
python scripts/validate_dataset.py --csv data/raw/Nuevo_final_df.csv --ncols 491
```

Both commands should report **28 passed, 0 failed**.

---

## Column organisation

Each row corresponds to one participant × one word recording session.

### Identity columns (3)

| Column | Type | Description |
|--------|------|-------------|
| `id_subject` | int | Participant ID (1–39) |
| `id_word` | int | Word ID (1–40) |
| `Clase` | int | Target label: 0 = correct, 1 = mispronounced |

### Acoustic features (84 columns)

Columns `duration` … `zero_crossing_rate`.
Extracted from the audio signal using openSMILE / Praat-style feature sets.
All participants × all words; no missing values by sentinel convention.

### EMG features — 4 channels × 99 columns each = 396 columns

| Channel | Column pattern | Valid subjects | Valid rows |
|---------|---------------|---------------|-----------|
| CH1 | `*_CH1*` | 16 | 640 |
| CH2 | `*_CH2*` | 31 | 1,238 |
| CH3 | `*_CH3*` | 19 | 760 |
| CH4 | `*_CH4*` | 22 | 880 |

**Sentinel value:** `888888` marks EMG data that was unavailable for a given
participant–word combination. The ML pipeline excludes rows where *any* column
of the target channel contains this sentinel before fitting/evaluating any model.

**Known data quality notes:**
- 195 NaN values exist across EMG columns (not sentinel).
- 32 ±Inf values are present exclusively in `lpc1`–`lpc16` EMG columns.
- The column `aqpq5_shimmer` is a known typographic variant of `apq5_shimmer`
  — this name is preserved as-is to maintain exact SHA-256 reproducibility.

---

## Phoneme structure

The 40 words are partitioned into 4 phoneme contrast categories:

| Phoneme | Word IDs | Class 0 | Class 1 |
|---------|----------|---------|---------|
| /t/ | 1,3,5,7,9,12,14,16,18,20 | 331 | 59 |
| /θ/ | 2,4,6,8,10,11,13,15,17,19 | 230 | 159 |
| /d/ | 21,23,25,27,29,32,34,36,38,40 | 291 | 99 |
| /ð/ | 22,24,26,28,30,31,33,35,37,39 | 161 | 228 |

The mapping is available in `metadata/phoneme_mapping.csv`.
The `english_word` column in that file is **UNVERIFIED** — the supplement
tables list the words but do not include a `id_word` numbering column.

---

## Experimental scenarios

The pipeline defines **20 ML scenarios** combining modality and channel:

| Code | Modality | Channel filter |
|------|----------|---------------|
| `A` | Acoustic only | All 1,558 rows |
| `EMG_CH1` | EMG CH1 only | 640 rows (16 subjects) |
| `EMG_CH2` | EMG CH2 only | 1,238 rows (31 subjects) |
| `EMG_CH3` | EMG CH3 only | 760 rows (19 subjects) |
| `EMG_CH4` | EMG CH4 only | 880 rows (22 subjects) |
| `A_CH1` | Acoustic + EMG CH1 | 640 rows |
| `A_CH2` | Acoustic + EMG CH2 | 1,238 rows |
| `A_CH3` | Acoustic + EMG CH3 | 760 rows |
| `A_CH4` | Acoustic + EMG CH4 | 880 rows |
| `A_CH{1–4}_{t,theta,d,eth}` | Acoustic + EMG, per phoneme | varies |

(plus phoneme-specific sub-scenarios, 20 total — see `docs/REPRODUCIBILITY.md`)

---

## License and ethics

> **Pending IRB and data sharing agreement documentation before public release.**
> See `docs/DATA_RELEASE_REVIEW.md` for the complete release checklist.

The raw biometric (EMG + audio) recordings are **not** included in this
repository. This dataset contains only derived tabular features.
