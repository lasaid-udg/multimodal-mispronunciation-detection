# Dataset

This directory contains the dataset accompanying the paper:

> **A Multimodal Machine Learning Approach to English Mispronunciation Detection
> Using Acoustic and Surface Electromyographic Signals**

---

## Files included

```
data/
├── derived/
│   ├── public_dataset.csv              # Feature dataset (1,558 rows × 483 columns)
│   └── public_dataset_manifest.json    # SHA-256 checksums
└── metadata/
    ├── data_dictionary.csv             # Column descriptions (CSV format)
    ├── data_dictionary.md              # Column descriptions (readable format)
    ├── schema.json                     # Column types and value ranges
    ├── dataset_statistics.json         # Summary statistics per column
    └── phoneme_mapping.csv             # Word ID → phoneme category (40 words)
```

---

## The dataset: `derived/public_dataset.csv`

| Property | Value |
|----------|-------|
| Rows | 1,558 (one per participant × word recording) |
| Columns | 483 |
| SHA-256 | `89EB58339E77F6C92EB949474F2338D520C3BF627B0F0853A43CD0F00EE1EA43` |
| Participants | 39 |
| Words | 40 (10 per phoneme contrast: /t/, /θ/, /d/, /ð/) |
| Class 0 — correct pronunciation | 1,013 rows |
| Class 1 — mispronunciation (positive class) | 545 rows |

### Column structure

Each row is one participant × one word recording.

| Group | Columns | Description |
|-------|---------|-------------|
| `id_subject` | 1 | Participant ID (integer, 1–39) |
| `id_word` | 1 | Word ID (integer, 1–40) |
| `Clase` | 1 | Label: 0 = correct, 1 = mispronounced |
| Acoustic features | 84 | Extracted from the audio signal |
| EMG CH1 features | 99 | Cricothyroid muscle |
| EMG CH2 features | 99 | Anterior belly of the digastric muscle |
| EMG CH3 features | 99 | Masseter muscle |
| EMG CH4 features | 99 | Zygomaticus major muscle |

### EMG channel subsets

Not all participants have usable EMG recordings for every channel.
The pipeline automatically selects the appropriate subset for each scenario.

| Channel | Usable participants | Usable rows |
|---------|-------------------|-------------|
| CH1 | 16 | 640 |
| CH2 | 31 | 1,238 |
| CH3 | 19 | 760 |
| CH4 | 22 | 880 |

### Special values

- **`888888`** — EMG data was unavailable for that participant–word combination.
  The pipeline excludes any row where the target channel contains this value.
- **`NaN`** — 195 values across EMG columns; handled by median imputation inside
  each training fold.
- **`±Inf`** — 32 values in `lpc1`–`lpc16` EMG columns; replaced with `NaN`
  before imputation.
- **`aqpq5_shimmer`** — known typographic variant of `apq5_shimmer`; the name is
  preserved as-is in this dataset.

---

## Phoneme structure

The 40 words are divided into 4 non-overlapping phoneme contrast groups:

| Phoneme | Word IDs | Correct | Mispronounced |
|---------|----------|---------|--------------|
| /t/ | 1,3,5,7,9,12,14,16,18,20 | 331 | 59 |
| /θ/ | 2,4,6,8,10,11,13,15,17,19 | 230 | 159 |
| /d/ | 21,23,25,27,29,32,34,36,38,40 | 291 | 99 |
| /ð/ | 22,24,26,28,30,31,33,35,37,39 | 161 | 228 |

The mapping from word ID to English word is available in
`metadata/phoneme_mapping.csv` and in Supplementary Tables S-1 and S-2 of the paper.

---

## Experimental scenarios

The pipeline uses this dataset across 20 scenarios (5 phoneme groups × 4 EMG channels),
each comparing two modalities:

| Modality | Features |
|----------|----------|
| `A` | Acoustic features only (84 columns) |
| `A_CH{1–4}` | Acoustic + EMG for that channel (84 + 99 = 183 columns) |

---

## Obtaining the original data

The pipeline can also run from the original 491-column file (`Nuevo_final_df.csv`),
which additionally contains 8 auxiliary ordering/timing columns not used by the ML pipeline.
To obtain it, contact the corresponding author.

---

## Verifying the dataset

```bash
md-emg validate-data
```

Expected output: `28 passed, 0 failed`.
