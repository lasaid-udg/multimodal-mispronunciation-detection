from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from .metrics import compute_metrics

DEFAULT_METRICS = [
    "balanced_accuracy", "mcc", "roc_auc", "average_precision",
    "sensitivity", "specificity", "f1_error", "accuracy",
]


def _load_predictions(run_dir: Path) -> pd.DataFrame:
    pq = run_dir / "predictions.parquet"
    csv = run_dir / "predictions.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"No predictions.parquet/csv found in {run_dir}")


def _scenario_modalities(preds: pd.DataFrame) -> tuple[str, str]:
    mods = sorted(preds["modality"].dropna().unique().tolist())
    if "A" not in mods:
        raise ValueError("Expected Audio-only modality 'A' in predictions.")
    multi = [m for m in mods if m != "A"]
    if len(multi) != 1:
        raise ValueError(f"Expected exactly one multimodal modality besides A; found {mods}")
    return "A", multi[0]


@dataclass(frozen=True)
class GroupArrays:
    y_true: np.ndarray
    y_pred: np.ndarray
    score: np.ndarray | None
    subject_pos: np.ndarray


@dataclass(frozen=True)
class AnalysisCache:
    subjects: np.ndarray
    repeats: list[int]
    models: list[str]
    modalities: tuple[str, str]
    groups: dict[tuple[str, str, int], GroupArrays]


def _build_cache(preds: pd.DataFrame, a_mod: str, b_mod: str) -> AnalysisCache:
    subjects = np.array(sorted(preds["id_subject"].unique()))
    sid_to_pos = {sid: i for i, sid in enumerate(subjects.tolist())}
    repeats = sorted(int(x) for x in preds["repeat"].unique())
    models = sorted(str(x) for x in preds["model"].unique())

    groups: dict[tuple[str, str, int], GroupArrays] = {}
    for (model, modality, repeat), g in preds.groupby(["model", "modality", "repeat"], sort=False):
        g = g.sort_values(["id_subject", "record_id"] if "record_id" in g.columns else ["id_subject"])
        score = g["score_error"].to_numpy(dtype=float) if "score_error" in g.columns else None
        groups[(str(model), str(modality), int(repeat))] = GroupArrays(
            y_true=g["y_true"].to_numpy(dtype=np.int8),
            y_pred=g["y_pred"].to_numpy(dtype=np.int8),
            score=score,
            subject_pos=np.fromiter((sid_to_pos[s] for s in g["id_subject"].tolist()), dtype=np.int32, count=len(g)),
        )
    return AnalysisCache(subjects, repeats, models, (a_mod, b_mod), groups)


def _metric_subset(y_true: np.ndarray, y_pred: np.ndarray, score: np.ndarray | None, metrics: Iterable[str], weights: np.ndarray | None = None) -> dict[str, float]:
    """Compute only requested metrics, optionally with cluster multiplicity weights."""
    metrics = set(metrics)
    out: dict[str, float] = {}

    # Drop zero-weight rows early. This keeps sklearn's class checks intuitive.
    if weights is not None:
        mask = weights > 0
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        score = score[mask] if score is not None else None
        weights = weights[mask]

    if len(y_true) == 0:
        return {m: np.nan for m in metrics}

    need_cm = bool(metrics & {"balanced_accuracy", "mcc", "sensitivity", "specificity", "f1_error", "accuracy"})
    if need_cm:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1], sample_weight=weights)
        tn, fp, fn, tp = cm.ravel().astype(float)
        total = tn + fp + fn + tp
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        if "sensitivity" in metrics:
            out["sensitivity"] = float(sens)
        if "specificity" in metrics:
            out["specificity"] = float(spec)
        if "balanced_accuracy" in metrics:
            out["balanced_accuracy"] = float((sens + spec) / 2.0) if np.isfinite(sens) and np.isfinite(spec) else np.nan
        if "accuracy" in metrics:
            out["accuracy"] = float((tp + tn) / total) if total else np.nan
        if "f1_error" in metrics:
            denom = 2 * tp + fp + fn
            out["f1_error"] = float((2 * tp) / denom) if denom else 0.0
        if "mcc" in metrics:
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            out["mcc"] = float(((tp * tn) - (fp * fn)) / denom) if denom else 0.0

    if "roc_auc" in metrics:
        if score is None or len(np.unique(y_true)) < 2:
            out["roc_auc"] = np.nan
        else:
            out["roc_auc"] = float(roc_auc_score(y_true, score, sample_weight=weights))
    if "average_precision" in metrics:
        if score is None or len(np.unique(y_true)) < 2:
            out["average_precision"] = np.nan
        else:
            out["average_precision"] = float(average_precision_score(y_true, score, sample_weight=weights))

    return {m: float(out.get(m, np.nan)) for m in metrics}


def _mean_dict(dicts: list[dict[str, float]], metrics: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in metrics:
        arr = np.asarray([d.get(metric, np.nan) for d in dicts], dtype=float)
        out[metric] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
    return out


def _observed_group_metrics(cache: AnalysisCache, metrics: list[str]) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for model in cache.models:
        for modality in cache.modalities:
            per_repeat = []
            for repeat in cache.repeats:
                g = cache.groups[(model, modality, repeat)]
                per_repeat.append(_metric_subset(g.y_true, g.y_pred, g.score, metrics))
            out[(model, modality)] = _mean_dict(per_repeat, metrics)
    return out


def _bootstrap_one(cache: AnalysisCache, multiplicities: np.ndarray, metrics: list[str]) -> tuple[dict[tuple[str, str, str], float], dict[tuple[str, str], float]]:
    vals: dict[tuple[str, str, str], float] = {}
    deltas: dict[tuple[str, str], float] = {}
    a_mod, b_mod = cache.modalities

    for model in cache.models:
        modality_metrics: dict[str, dict[str, float]] = {}
        for modality in cache.modalities:
            per_repeat = []
            for repeat in cache.repeats:
                g = cache.groups[(model, modality, repeat)]
                row_w = multiplicities[g.subject_pos]
                per_repeat.append(_metric_subset(g.y_true, g.y_pred, g.score, metrics, row_w))
            mm = _mean_dict(per_repeat, metrics)
            modality_metrics[modality] = mm
            for metric, value in mm.items():
                vals[(model, modality, metric)] = value
        for metric in metrics:
            deltas[(model, metric)] = modality_metrics[b_mod][metric] - modality_metrics[a_mod][metric]
    return vals, deltas


def _paired_group(cache: AnalysisCache, model: str, repeat: int) -> tuple[GroupArrays, GroupArrays]:
    a_mod, b_mod = cache.modalities
    ga = cache.groups[(model, a_mod, repeat)]
    gb = cache.groups[(model, b_mod, repeat)]
    if not (
        np.array_equal(ga.y_true, gb.y_true)
        and np.array_equal(ga.subject_pos, gb.subject_pos)
        and len(ga.y_pred) == len(gb.y_pred)
    ):
        raise ValueError(f"A and multimodal predictions are not row-aligned for model={model}, repeat={repeat}")
    return ga, gb


def _permutation_one(cache: AnalysisCache, swap_subject: np.ndarray, metrics: list[str]) -> dict[tuple[str, str], float]:
    a_mod, b_mod = cache.modalities
    result: dict[tuple[str, str], float] = {}
    for model in cache.models:
        a_repeat: list[dict[str, float]] = []
        b_repeat: list[dict[str, float]] = []
        for repeat in cache.repeats:
            ga, gb = _paired_group(cache, model, repeat)
            swap_rows = swap_subject[ga.subject_pos]
            a_pred = np.where(swap_rows, gb.y_pred, ga.y_pred)
            b_pred = np.where(swap_rows, ga.y_pred, gb.y_pred)
            if ga.score is not None and gb.score is not None:
                a_score = np.where(swap_rows, gb.score, ga.score)
                b_score = np.where(swap_rows, ga.score, gb.score)
            else:
                a_score = b_score = None
            a_repeat.append(_metric_subset(ga.y_true, a_pred, a_score, metrics))
            b_repeat.append(_metric_subset(gb.y_true, b_pred, b_score, metrics))
        am = _mean_dict(a_repeat, metrics)
        bm = _mean_dict(b_repeat, metrics)
        for metric in metrics:
            result[(model, metric)] = bm[metric] - am[metric]
    return result


def _progress_marks(total: int) -> set[int]:
    if total <= 10:
        return set(range(1, total + 1))
    return {max(1, int(round(total * p / 10))) for p in range(1, 11)}



def _parallel_batches(total: int, worker, *, n_jobs: int, label: str, verbose: bool, batch: int | None = None):
    if total <= 0:
        return []
    batch = batch or max(10, min(100, total // 10 if total >= 10 else total))
    out = []
    for start in range(0, total, batch):
        stop = min(total, start + batch)
        out.extend(Parallel(n_jobs=n_jobs, prefer="threads", batch_size="auto")(
            delayed(worker)(i) for i in range(start, stop)
        ))
        if verbose:
            pct = 100.0 * stop / total
            print(f"[ANALYZE] {label}: {stop}/{total} ({pct:.0f}%)")
    return out

def analyze_run(
    run_dir: str | Path,
    *,
    bootstrap: int = 2000,
    permutations: int = 2000,
    seed: int = 2026,
    metrics: list[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Path:
    """Fast post-hoc repeated-OOF analysis without retraining.

    The statistical definitions are unchanged from v3.2:
    - participant-cluster bootstrap (sampling complete participants with replacement)
    - paired participant-level modality permutation test
    - metrics are computed within each OOF repeat and then averaged across repeats

    This implementation avoids pandas work inside resampling loops, computes each
    metric set once per group, and can parallelize resamples with joblib threads.
    """
    run_dir = Path(run_dir).resolve()
    if bootstrap < 1:
        raise ValueError("bootstrap must be >= 1")
    if permutations < 0:
        raise ValueError("permutations must be >= 0")
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be 0")
    metrics = metrics or DEFAULT_METRICS

    preds = _load_predictions(run_dir)
    required = {"id_subject", "repeat", "modality", "model", "y_true", "y_pred"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {sorted(missing)}")

    a_mod, b_mod = _scenario_modalities(preds)
    cache = _build_cache(preds, a_mod, b_mod)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)

    if verbose:
        print(f"[ANALYZE] subjects={len(cache.subjects)} | repeats={len(cache.repeats)} | models={len(cache.models)}")
        print(f"[ANALYZE] bootstrap={bootstrap} | permutations={permutations} | n_jobs={n_jobs}")
        print(f"[ANALYZE] metrics={', '.join(metrics)}")

    # 1) Metrics for every OOF repeat.
    repeat_rows = []
    for model in cache.models:
        for modality in cache.modalities:
            for repeat in cache.repeats:
                g = cache.groups[(model, modality, repeat)]
                score = g.score
                row = {"repeat": int(repeat), "modality": modality, "model": model}
                row.update(compute_metrics(g.y_true, g.y_pred, score))
                repeat_rows.append(row)
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(out_dir / "repeat_metrics.csv", index=False)

    summary_rows = []
    for (modality, model), g in repeat_df.groupby(["modality", "model"], sort=True):
        row = {"modality": modality, "model": model, "n_repeats": int(len(g))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(g[metric].mean())
            row[f"{metric}_sd"] = float(g[metric].std(ddof=1)) if len(g) > 1 else np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(out_dir / "repeat_summary.csv", index=False)

    observed_mm = _observed_group_metrics(cache, metrics)
    observed = {
        (model, modality, metric): observed_mm[(model, modality)][metric]
        for model in cache.models for modality in cache.modalities for metric in metrics
    }
    obs_delta = {
        (model, metric): observed[(model, b_mod, metric)] - observed[(model, a_mod, metric)]
        for model in cache.models for metric in metrics
    }

    # 2) Participant-level cluster bootstrap.
    rng = np.random.default_rng(seed)
    n_subjects = len(cache.subjects)
    # Multinomial multiplicities are exactly equivalent to drawing n_subjects
    # participant IDs with replacement and counting each participant's copies.
    boot_mult = rng.multinomial(n_subjects, np.full(n_subjects, 1.0 / n_subjects), size=bootstrap).astype(np.int16)

    if verbose:
        print("[ANALYZE] Bootstrap started...")
    boot_results = _parallel_batches(
        bootstrap,
        lambda i: _bootstrap_one(cache, boot_mult[i], metrics),
        n_jobs=n_jobs, label="Bootstrap", verbose=verbose,
    )

    boot_values = {(model, modality, metric): [] for model in cache.models for modality in cache.modalities for metric in metrics}
    delta_values = {(model, metric): [] for model in cache.models for metric in metrics}
    for vals, deltas in boot_results:
        for key, value in vals.items():
            boot_values[key].append(value)
        for key, value in deltas.items():
            delta_values[key].append(value)

    ci_rows = []
    for (model, modality, metric), vals in boot_values.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        ci_rows.append({
            "model": model,
            "modality": modality,
            "metric": metric,
            "estimate": observed[(model, modality, metric)],
            "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "n_bootstrap_valid": int(len(arr)),
        })
    pd.DataFrame(ci_rows).to_csv(out_dir / "bootstrap_ci.csv", index=False)

    # 3) Paired participant-level modality permutation test.
    perm_extreme = {(model, metric): 0 for model in cache.models for metric in metrics}
    if permutations:
        # Independent 50/50 swap decision for each participant in every permutation.
        swap_matrix = rng.integers(0, 2, size=(permutations, n_subjects), dtype=np.int8).astype(bool)
        if verbose:
            print("[ANALYZE] Permutation test started...")
        perm_results = _parallel_batches(
            permutations,
            lambda i: _permutation_one(cache, swap_matrix[i], metrics),
            n_jobs=n_jobs, label="Permutation", verbose=verbose,
        )
        for result in perm_results:
            for key, delta in result.items():
                if np.isfinite(delta) and abs(delta) >= abs(obs_delta[key]):
                    perm_extreme[key] += 1

    delta_rows = []
    for (model, metric), vals in delta_values.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        p = ((perm_extreme[(model, metric)] + 1) / (permutations + 1)) if permutations else np.nan
        delta_rows.append({
            "model": model,
            "metric": metric,
            "comparison": f"{b_mod} - {a_mod}",
            "delta_estimate": obs_delta[(model, metric)],
            "delta_ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
            "delta_ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
            "permutation_p": float(p) if np.isfinite(p) else np.nan,
            "n_bootstrap_valid": int(len(arr)),
            "n_permutations": int(permutations),
        })
    pd.DataFrame(delta_rows).to_csv(out_dir / "paired_deltas.csv", index=False)

    meta = {
        "run_dir": str(run_dir),
        "audio_modality": a_mod,
        "multimodal_modality": b_mod,
        "n_subjects": int(n_subjects),
        "n_repeats": int(len(cache.repeats)),
        "bootstrap": int(bootstrap),
        "permutations": int(permutations),
        "seed": int(seed),
        "n_jobs": int(n_jobs),
        "implementation": "fast-v3.2.1",
        "metrics": metrics,
        "note": "Bootstrap resamples participants; each resample computes metrics within each OOF repeat then averages across repeats. Deltas are paired A+EMG minus A. Multinomial participant multiplicities are equivalent to explicit cluster resampling with replacement.",
    }
    with (out_dir / "analysis_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return out_dir
