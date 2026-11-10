"""
scripts/validate_dataset.py
============================
Automated integrity tests for the public derived dataset.

Tests FAIL explicitly when scientific data has been altered.
Tests are designed to detect silent modifications to labels, predictors,
participant groups, channel exclusion rules, and sentinel values.

Usage
-----
From project root (md_emg_refactor/):
    # Validate the 483-column public dataset (default)
    python scripts/validate_dataset.py

    # Validate the original 491-column CSV
    python scripts/validate_dataset.py --csv data/raw/Nuevo_final_df.csv --ncols 491

    # Validate with explicit SHA-256
    python scripts/validate_dataset.py --expected-sha256 89EB58...

Exit codes
----------
    0 = all tests passed
    1 = one or more tests failed
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Expected values (from manuscript and verified against CSV) ───────────────
EXPECTED = {
    "public_csv":  "data/derived/public_dataset.csv",
    "original_csv": "data/raw/Nuevo_final_df.csv",
    "sha256_public":   "E47AE2E5510F29A290705DD0718198E7B3801CDD670AD960095D6569BC36C028",
    "sha256_original": "98499BE472A8626F88B6330850ECBAB2218E9BE69B26F79D6A2502A789FFF2E6",
    "n_rows":    1558,
    "n_cols_public":   483,
    "n_cols_original": 491,
    "n_subjects": 39,
    "n_words":    40,
    "n_class0":  1013,
    "n_class1":   545,
    "positive_class": 1,
    "channel_clean_subjects": {"ch1": 16, "ch2": 31, "ch3": 19, "ch4": 22},
    "channel_clean_rows":     {"ch1": 640, "ch2": 1238, "ch3": 760, "ch4": 880},
    "n_audio_predictors": 84,
    "n_emg_predictors_per_channel": 99,
    "sentinel": 888888,
}

PHONEME_WORDS = {
    "t":     [1,3,5,7,9,12,14,16,18,20],
    "theta": [2,4,6,8,10,11,13,15,17,19],
    "d":     [21,23,25,27,29,32,34,36,38,40],
    "eth":   [22,24,26,28,30,31,33,35,37,39],
}

PHONEME_CLASS_COUNTS = {
    "t":     (331, 59),
    "theta": (230, 159),
    "d":     (291, 99),
    "eth":   (161, 228),
}


# ─── Test harness ─────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def ok(self, name: str, msg: str = ""):
        marker = f"  [PASS] {name}"
        if msg:
            marker += f": {msg}"
        print(marker)
        self.passed.append(name)

    def fail(self, name: str, msg: str):
        marker = f"  [FAIL] {name}: {msg}"
        print(marker)
        self.failed.append(name)

    def summary(self) -> int:
        print()
        print(f"Results: {len(self.passed)} passed, {len(self.failed)} failed")
        return 1 if self.failed else 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def get_channel_cols(df: pd.DataFrame, ch: int) -> tuple[list, list, list]:
    """Returns (all_ch_cols, predictor_cols, aux_cols)."""
    patt = re.compile(rf"(^CH{ch}_|_CH{ch}$|_CH{ch}_)", re.IGNORECASE)
    all_ch = [c for c in df.columns if patt.search(c)]
    aux = [c for c in all_ch if c.lower().startswith("order_ch") or c.lower().startswith("time_ch")]
    pred = [c for c in all_ch if c not in aux]
    return all_ch, pred, aux


def run_tests(df: pd.DataFrame, r: TestResult, expected_ncols: int, expected_sha256: str | None,
              csv_path: Path) -> None:
    # ── T1: SHA-256 ──────────────────────────────────────────────────────────
    if expected_sha256:
        actual_sha = sha256_file(csv_path)
        if actual_sha == expected_sha256.upper():
            r.ok("T01_sha256", f"{actual_sha[:16]}...")
        else:
            r.fail("T01_sha256", f"expected {expected_sha256[:16]}... got {actual_sha[:16]}...")
    else:
        print("  [SKIP] T01_sha256: no expected SHA-256 provided")

    # ── T2: Row count ─────────────────────────────────────────────────────────
    if len(df) == EXPECTED["n_rows"]:
        r.ok("T02_row_count", f"{len(df)} rows")
    else:
        r.fail("T02_row_count", f"expected {EXPECTED['n_rows']}, got {len(df)}")

    # ── T3: Column count ──────────────────────────────────────────────────────
    if df.shape[1] == expected_ncols:
        r.ok("T03_col_count", f"{df.shape[1]} cols")
    else:
        r.fail("T03_col_count", f"expected {expected_ncols}, got {df.shape[1]}")

    # ── T4: 39 unique participants ─────────────────────────────────────────────
    n_subj = df["id_subject"].nunique()
    if n_subj == EXPECTED["n_subjects"]:
        r.ok("T04_subject_count", f"{n_subj} subjects")
    else:
        r.fail("T04_subject_count", f"expected {EXPECTED['n_subjects']}, got {n_subj}")

    # ── T5: 40 unique words ────────────────────────────────────────────────────
    n_words = df["id_word"].nunique()
    if n_words == EXPECTED["n_words"]:
        r.ok("T05_word_count", f"{n_words} words")
    else:
        r.fail("T05_word_count", f"expected {EXPECTED['n_words']}, got {n_words}")

    # ── T6: Class distribution ────────────────────────────────────────────────
    n0 = int((df["Clase"] == 0).sum())
    n1 = int((df["Clase"] == 1).sum())
    if n0 == EXPECTED["n_class0"] and n1 == EXPECTED["n_class1"]:
        r.ok("T06_class_distribution", f"class0={n0}, class1={n1}")
    else:
        r.fail("T06_class_distribution", f"expected 0:{EXPECTED['n_class0']}/1:{EXPECTED['n_class1']}, got 0:{n0}/1:{n1}")

    # ── T7: Target values only 0 and 1 ───────────────────────────────────────
    invalid_target = df["Clase"].loc[~df["Clase"].isin([0, 1])]
    if len(invalid_target) == 0:
        r.ok("T07_target_values_binary", "all values are 0 or 1")
    else:
        r.fail("T07_target_values_binary", f"{len(invalid_target)} non-binary target values found")

    # ── T8: No duplicate (subject, word) pairs ────────────────────────────────
    dups = df.duplicated(subset=["id_subject", "id_word"]).sum()
    if dups == 0:
        r.ok("T08_no_duplicate_pairs", "0 duplicate (id_subject, id_word) pairs")
    else:
        r.fail("T08_no_duplicate_pairs", f"{dups} duplicate (id_subject, id_word) pairs found")

    # ── T9: Acoustic feature count ────────────────────────────────────────────
    cols = df.columns.tolist()
    if "duration" in cols and "zero_crossing_rate" in cols:
        i0 = cols.index("duration")
        i1 = cols.index("zero_crossing_rate")
        n_audio = i1 - i0 + 1
        if n_audio == EXPECTED["n_audio_predictors"]:
            r.ok("T09_acoustic_predictor_count", f"{n_audio} columns")
        else:
            r.fail("T09_acoustic_predictor_count", f"expected {EXPECTED['n_audio_predictors']}, got {n_audio}")
    else:
        r.fail("T09_acoustic_predictor_count", "anchor columns duration/zero_crossing_rate not found")

    # ── T10: EMG predictor count per channel ──────────────────────────────────
    for ch in range(1, 5):
        _, pred, _ = get_channel_cols(df, ch)
        expected_pred = EXPECTED["n_emg_predictors_per_channel"]
        if len(pred) == expected_pred:
            r.ok(f"T10_emg_ch{ch}_predictor_count", f"{len(pred)} predictors")
        else:
            r.fail(f"T10_emg_ch{ch}_predictor_count", f"expected {expected_pred}, got {len(pred)}")

    # ── T11: Channel sentinel exclusion produces expected subject counts ───────
    for ch in range(1, 5):
        all_ch, _, _ = get_channel_cols(df, ch)
        if not all_ch:
            r.fail(f"T11_channel_ch{ch}_clean_subjects", f"no CH{ch} columns found")
            continue
        bad = (df[all_ch] == EXPECTED["sentinel"]).any(axis=1)
        clean_subjects = df.loc[~bad, "id_subject"].nunique()
        clean_rows     = int((~bad).sum())
        exp_subj = EXPECTED["channel_clean_subjects"][f"ch{ch}"]
        exp_rows = EXPECTED["channel_clean_rows"][f"ch{ch}"]
        if clean_subjects == exp_subj and clean_rows == exp_rows:
            r.ok(f"T11_channel_ch{ch}_exclusion", f"{clean_subjects} subjects, {clean_rows} rows")
        else:
            r.fail(f"T11_channel_ch{ch}_exclusion",
                   f"expected subj={exp_subj}/rows={exp_rows}, got subj={clean_subjects}/rows={clean_rows}")

    # ── T12: Sentinel not treated as measurement (NaN/inf/0 are OK, 888888 is not a valid EMG value) ─
    sentinel_in_audio = 0
    if "duration" in cols and "zero_crossing_rate" in cols:
        i0 = cols.index("duration"); i1 = cols.index("zero_crossing_rate")
        audio_cols = cols[i0:i1+1]
        sentinel_in_audio = int((df[audio_cols] == EXPECTED["sentinel"]).sum().sum())
    if sentinel_in_audio == 0:
        r.ok("T12_sentinel_not_in_acoustic_cols", "no 888888 values in acoustic predictor columns")
    else:
        r.fail("T12_sentinel_not_in_acoustic_cols", f"{sentinel_in_audio} sentinel values found in acoustic columns")

    # ── T13: Phoneme word ID partition is complete and disjoint ───────────────
    all_words = set(range(1, 41))
    all_assigned = set()
    overlap = False
    for ph, words in PHONEME_WORDS.items():
        wset = set(words)
        if all_assigned & wset:
            overlap = True
        all_assigned |= wset
    if not overlap and all_assigned == all_words and len(all_assigned) == 40:
        r.ok("T13_phoneme_word_partition", "40 words, 4 phonemes, no overlap, complete coverage")
    else:
        r.fail("T13_phoneme_word_partition", f"overlap={overlap}, coverage={len(all_assigned)}/40")

    # ── T14: Phoneme class distributions match manuscript ─────────────────────
    for ph, words in PHONEME_WORDS.items():
        subset = df.loc[df["id_word"].isin(words)]
        actual_0 = int((subset["Clase"] == 0).sum())
        actual_1 = int((subset["Clase"] == 1).sum())
        exp_0, exp_1 = PHONEME_CLASS_COUNTS[ph]
        if actual_0 == exp_0 and actual_1 == exp_1:
            r.ok(f"T14_phoneme_{ph}_class_distribution", f"class0={actual_0}, class1={actual_1}")
        else:
            r.fail(f"T14_phoneme_{ph}_class_distribution",
                   f"expected 0:{exp_0}/1:{exp_1}, got 0:{actual_0}/1:{actual_1}")

    # ── T15: A and A+EMG have same records (spot-check CH2, all phonemes) ─────
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from md_emg.config import load_config
        from md_emg.data import build_scenario
        cfg_path = Path(__file__).parent.parent / "configs" / "paper.yaml"
        if cfg_path.exists():
            cfg = load_config(str(cfg_path))
            cfg["scenario"]["channel"] = "ch2"
            cfg["scenario"]["phoneme"] = "all"
            Xs, y, groups, meta, _ = build_scenario(df, cfg)
            a_cols   = set(Xs["A"].columns)
            b_cols   = set(Xs["A_CH2"].columns)
            # A columns must be subset of A_CH2 columns
            a_only = a_cols - b_cols  # should be empty: A subset of A_CH2
            if not a_only:
                r.ok("T15_A_subset_of_A_CH2_columns", f"A={len(a_cols)} cols subset-of A_CH2={len(b_cols)} cols")
            else:
                r.fail("T15_A_subset_of_A_CH2_columns", f"A has {len(a_only)} cols not in A_CH2")
        else:
            print("  [SKIP] T15: configs/paper.yaml not found")
    except Exception as e:
        r.fail("T15_A_subset_of_A_CH2_columns", f"exception: {e}")

    # ── T16: No inf in sentinel positions (sentinel cells contain exactly 888888) ─
    for ch in range(1, 5):
        all_ch, pred, _ = get_channel_cols(df, ch)
        if not pred:
            continue
        bad_mask = (df[pred] == EXPECTED["sentinel"]).any(axis=1)
        sentinel_rows = df.loc[bad_mask, pred]
        # In rows where ANY column is 888888, ALL non-sentinel numeric values should not be inf
        non_sent = sentinel_rows.replace(EXPECTED["sentinel"], np.nan)
        n_inf = int(np.isinf(pd.to_numeric(non_sent.stack(), errors="coerce")).sum())
        if n_inf == 0:
            r.ok(f"T16_ch{ch}_no_inf_in_non_sentinel_emg", "no inf values alongside sentinels")
        else:
            r.fail(f"T16_ch{ch}_no_inf_in_non_sentinel_emg", f"{n_inf} inf values found in non-sentinel EMG positions")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the public derived dataset.")
    parser.add_argument("--csv", type=Path,
                        default=Path("data/derived/public_dataset.csv"),
                        help="Path to CSV to validate")
    parser.add_argument("--ncols", type=int, default=483,
                        help="Expected number of columns (483 for public, 491 for original)")
    parser.add_argument("--expected-sha256", type=str, default=None,
                        help="Expected SHA-256 (uppercase hex). If omitted, uses preset for public/original.")
    args = parser.parse_args()

    # Auto-select expected SHA-256 if not provided
    if args.expected_sha256 is None:
        if args.ncols == 483:
            args.expected_sha256 = EXPECTED["sha256_public"]
        elif args.ncols == 491:
            args.expected_sha256 = EXPECTED["sha256_original"]

    print(f"Validating: {args.csv}  (expected cols={args.ncols})")
    print()

    if not args.csv.exists():
        print(f"ERROR: File not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    r  = TestResult()
    run_tests(df, r, args.ncols, args.expected_sha256, args.csv)
    return r.summary()


if __name__ == "__main__":
    sys.exit(main())
