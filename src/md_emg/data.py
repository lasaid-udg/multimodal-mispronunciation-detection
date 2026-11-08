from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_registry, predictor_columns

PHONEME_WORDS = {
    "all": None,
    "t": [1, 3, 5, 7, 9, 12, 14, 16, 18, 20],
    "theta": [2, 4, 6, 8, 10, 11, 13, 15, 17, 19],
    "d": [21, 23, 25, 27, 29, 32, 34, 36, 38, 40],
    "eth": [22, 24, 26, 28, 30, 31, 33, 35, 37, 39],
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def make_record_id(df: pd.DataFrame, subject_col: str, word_col: str) -> pd.Series:
    s = pd.to_numeric(df[subject_col], errors="raise").astype(int).astype(str)
    w = pd.to_numeric(df[word_col], errors="raise").astype(int).astype(str)
    return "s" + s + "_w" + w


def audit_dataframe(df: pd.DataFrame, cfg: dict) -> dict:
    dcfg = cfg["data"]
    sc, wc, tc = dcfg["subject_col"], dcfg["word_col"], dcfg["target_col"]
    marker = dcfg["damage_marker"]
    registry = build_feature_registry(df.columns.tolist())
    rid = make_record_id(df, sc, wc)
    target = pd.to_numeric(df[tc], errors="coerce")
    out = {
        "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
        "n_subjects": int(df[sc].nunique()), "n_words": int(df[wc].nunique()),
        "class_counts": {str(int(k)): int(v) for k,v in target.value_counts(dropna=False).items() if pd.notna(k)},
        "invalid_target_count": int((~target.isin([0,1])).sum()),
        "duplicate_record_ids": int(rid.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "infinite_values": int(np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).sum()),
        "feature_counts": {k: len(v) for k,v in registry.items()},
        "channels": {}
    }
    for ch in ["ch1","ch2","ch3","ch4"]:
        cc = registry[ch]
        bad = (df[cc] == marker).any(axis=1)
        out["channels"][ch] = {
            "n_features": len(cc), "damaged_rows": int(bad.sum()), "clean_rows": int((~bad).sum()),
            "clean_subjects": int(df.loc[~bad, sc].nunique()), "damaged_subjects": int(df.loc[bad, sc].nunique())
        }
    return out


def build_scenario(df: pd.DataFrame, cfg: dict):
    dcfg, scfg = cfg["data"], cfg["scenario"]
    sc, wc, tc = dcfg["subject_col"], dcfg["word_col"], dcfg["target_col"]
    marker = dcfg["damage_marker"]
    channel = scfg["channel"].lower()
    phoneme = str(scfg.get("phoneme", "all")).lower()
    if phoneme not in PHONEME_WORDS:
        raise ValueError(f"Unsupported phoneme: {phoneme}. Use one of: {', '.join(PHONEME_WORDS)}")

    registry = build_feature_registry(df.columns.tolist())
    audio, emg = predictor_columns(registry, channel, scfg.get("exclude_order_time", True))
    channel_all = registry[channel]

    # Phoneme subsets are defined by the original study's id_word mapping.
    # Filter the requested phoneme first, then apply the channel-quality mask so
    # Audio and Audio+EMG remain perfectly row-matched within each scenario.
    if PHONEME_WORDS[phoneme] is None:
        phoneme_df = df
    else:
        phoneme_df = df.loc[df[wc].isin(PHONEME_WORDS[phoneme])]

    bad = (phoneme_df[channel_all] == marker).any(axis=1)
    use = phoneme_df.loc[~bad].copy()
    use["record_id"] = make_record_id(use, sc, wc)
    # Fixed deterministic cleanup only; learned imputation happens in Pipeline.
    use[audio + emg] = use[audio + emg].replace([np.inf, -np.inf], np.nan)
    X_audio = use[audio].apply(pd.to_numeric, errors="coerce")
    X_multi = use[audio + emg].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(use[tc], errors="raise").astype(int)
    groups = pd.to_numeric(use[sc], errors="raise").astype(int)
    meta = use[["record_id", sc, wc]].copy()
    return {"A": X_audio, f"A_{channel.upper()}": X_multi}, y, groups, meta, {
        "audio": audio,
        "emg": emg,
        "all_channel": channel_all,
        "phoneme": phoneme,
        "phoneme_word_ids": PHONEME_WORDS[phoneme],
        "rows_before_channel_filter": int(len(phoneme_df)),
        "rows_after_channel_filter": int(len(use)),
    }
