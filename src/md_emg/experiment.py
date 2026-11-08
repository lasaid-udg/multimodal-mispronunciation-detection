from __future__ import annotations

import json
import time
import warnings
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from .data import audit_dataframe, build_scenario, read_csv, sha256_file
from .metrics import compute_metrics
from .models import (
    build_model,
    continuous_score,
    estimator_diagnostics,
    selected_class_weight,
)
from .outputs import environment_manifest, make_run_dir, save_json, save_table, save_yaml
from .progress import RunReporter


def _resolve(base: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base / q).resolve()


def _candidate_count(grid) -> int:
    return len(list(ParameterGrid(grid)))


def _short_metric(m: dict) -> str:
    return (
        f"BA={m['balanced_accuracy']:.3f} | F1={m['f1_error']:.3f} | "
        f"Sens={m['sensitivity']:.3f} | Spec={m['specificity']:.3f} | MCC={m['mcc']:.3f}"
    )


def _scientific_config(cfg: dict) -> dict:
    """Subset that must not change when resuming an interrupted run."""
    keys = ["experiment", "data", "scenario", "cv", "models", "balance", "selection"]
    return {k: deepcopy(cfg.get(k)) for k in keys}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_table(base: Path) -> pd.DataFrame:
    if base.exists():
        return pd.read_parquet(base)
    csv = base.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(base)


def _unit_is_complete(run_dir: Path, stem: str) -> bool:
    pred_ok = (run_dir / "fold_outputs" / f"pred_{stem}.parquet").exists() or (run_dir / "fold_outputs" / f"pred_{stem}.csv").exists()
    return (
        pred_ok
        and (run_dir / "fold_outputs" / f"metric_{stem}.json").exists()
        and (run_dir / "fold_outputs" / f"params_{stem}.json").exists()
    )


def _prepare_run(cfg: dict, out_root: Path, command: str, data_path: Path,
                 resume_dir: Path | None, project_root: Path | None = None):
    clean_cfg = deepcopy(cfg)
    clean_cfg.pop("_config_path", None)

    if resume_dir is None:
        run_id, run_dir = make_run_dir(out_root, clean_cfg["experiment"]["name"])
        save_yaml(clean_cfg, run_dir / "config.yaml")
        manifest = environment_manifest(command, project_root=project_root)
        manifest.update({"run_id": run_id, "data_path": str(data_path), "data_sha256": sha256_file(data_path)})
        save_json(manifest, run_dir / "manifest.json")
        return run_id, run_dir, clean_cfg, False

    run_dir = Path(resume_dir).resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"Resume directory does not exist: {run_dir}")
    config_file = run_dir / "config.yaml"
    manifest_file = run_dir / "manifest.json"
    if not config_file.exists() or not manifest_file.exists():
        raise RuntimeError("Resume directory is missing config.yaml or manifest.json")

    with config_file.open("r", encoding="utf-8") as f:
        old_cfg = yaml.safe_load(f)
    if _scientific_config(old_cfg) != _scientific_config(clean_cfg):
        raise RuntimeError(
            "Scientific configuration differs from the interrupted run. "
            "Use the same config/seed/CV/models/scenario when resuming; runtime n_jobs/verbosity may differ."
        )

    old_manifest = _load_json(manifest_file)
    current_hash = sha256_file(data_path)
    if old_manifest.get("data_sha256") != current_hash:
        raise RuntimeError("Dataset hash differs from the interrupted run; refusing to resume.")

    run_id = old_manifest.get("run_id", run_dir.name)
    return run_id, run_dir, clean_cfg, True


def run_experiment(cfg: dict, command: str = "", resume_dir: Path | None = None) -> Path:
    config_path = Path(cfg.get("_config_path", ".")).resolve()
    project_root = config_path.parent.parent if config_path.name != "." else Path.cwd()
    data_path = _resolve(project_root, cfg["data"]["path"])
    out_root = _resolve(project_root, cfg["output"]["root"])

    run_id, run_dir, cfg, is_resume = _prepare_run(cfg, out_root, command, data_path, resume_dir,
                                                    project_root=project_root)
    reporter = RunReporter(run_dir / "logs" / "run.log", verbose=cfg["compute"].get("verbose", True))

    reporter.info(("Resuming" if is_resume else "Run") + f": {run_id}")
    reporter.info(f"Data: {data_path}")
    reporter.info(
        f"Scenario: phoneme={cfg['scenario'].get('phoneme', 'all')} | channel={cfg['scenario']['channel'].upper()} | "
        f"exclude_order_time={cfg['scenario'].get('exclude_order_time', True)}"
    )
    reporter.info(
        f"Target: {cfg['data']['target_col']} | positive class: 1 (mispronunciation) | "
        f"group variable: {cfg['data']['subject_col']}"
    )
    reporter.info(
        "Leakage protection: participant-disjoint outer CV=yes | participant-grouped inner CV=yes | "
        "preprocessing inside pipeline=yes"
    )
    reporter.info(
        f"CV protocol: outer={cfg['cv']['outer_folds']} folds | inner={cfg['cv']['inner_folds']} folds | "
        f"repeats={cfg['cv'].get('repeats', 1)} | shuffle={cfg['cv'].get('shuffle', True)} | "
        f"base_seed={cfg['experiment']['seed']}"
    )
    reporter.info(f"Selection metric: {cfg['selection']['scoring']}")
    reporter.info(f"Models: {', '.join(cfg['models']['enabled'])}")
    reporter.info(
        f"Class-weight mode: {cfg.get('balance', {}).get('class_weight', 'tune')} "
        f"(supported by svm/rf/dt/lr; knn/gnb/mlp remain unweighted)"
    )
    reporter.info(f"Parallel jobs for inner search: {cfg['compute'].get('n_jobs', 1)} | estimator threads limited to 1")

    # Audit-friendly model summary before any fitting starts.  Candidate counts
    # already include class_weight tuning when applicable.
    for model_name in cfg['models']['enabled']:
        _, preview_grid = build_model(model_name, cfg, int(cfg['experiment']['seed']))
        candidates = _candidate_count(preview_grid)
        m = cfg['models'][model_name]
        extras = []
        if model_name == 'mlp':
            extras.extend([
                f"max_iter={m.get('max_iter', 3000)}",
                f"early_stopping={m.get('early_stopping', True)}",
                f"n_iter_no_change={m.get('n_iter_no_change', 20)}",
            ])
        elif model_name == 'lr':
            extras.append(f"max_iter={m.get('max_iter', 2000)}")
        suffix = (" | " + " | ".join(extras)) if extras else ""
        reporter.info(
            f"Model {model_name.upper()}: {candidates} inner-search candidate(s){suffix} | "
            f"grid={json.dumps(m.get('grid', {}), ensure_ascii=False, default=str)}"
        )

    df = read_csv(data_path)
    audit = audit_dataframe(df, cfg)
    save_json(audit, run_dir / "audit.json")
    Xs, y, groups, meta, feature_info = build_scenario(df, cfg)
    save_json(feature_info, run_dir / "features.json")
    ph = cfg['scenario'].get('phoneme', 'all')
    word_ids = feature_info.get('phoneme_word_ids')
    reporter.info(
        f"Phoneme subset: {ph} | word_ids={word_ids if word_ids is not None else 'all'} | "
        f"rows_before_channel_filter={feature_info.get('rows_before_channel_filter')}"
    )
    reporter.info(
        f"Scenario {cfg['scenario']['channel'].upper()} clean: rows={len(y)} | subjects={groups.nunique()} | "
        f"class0={(y == 0).sum()} | class1={(y == 1).sum()}"
    )
    reporter.info("Features: " + " | ".join(f"{k}={v.shape[1]}" for k, v in Xs.items()))

    seed = int(cfg["experiment"]["seed"])
    outer_n = int(cfg["cv"]["outer_folds"])
    inner_n = int(cfg["cv"]["inner_folds"])
    repeats = int(cfg["cv"].get("repeats", 1))
    n_jobs = int(cfg["compute"].get("n_jobs", 1))
    scoring = cfg["selection"]["scoring"]

    all_split_rows = []
    pred_frames = []
    fold_metrics = []
    best_params = []

    total_units = repeats * len(Xs) * len(cfg["models"]["enabled"]) * outer_n
    unit_no = 0
    skipped = 0
    run_t0 = time.time()

    for repeat in range(1, repeats + 1):
        repeat_seed = seed + (repeat - 1) * 10000
        outer = StratifiedGroupKFold(
            n_splits=outer_n,
            shuffle=cfg["cv"].get("shuffle", True),
            random_state=repeat_seed,
        )
        split_list = list(outer.split(next(iter(Xs.values())), y, groups))
        reporter.step(f"Repeat {repeat}/{repeats}: built {len(split_list)} participant-grouped outer folds (seed={repeat_seed})")

        for fold, (tr, te) in enumerate(split_list, 1):
            train_sub = set(groups.iloc[tr]); test_sub = set(groups.iloc[te])
            if train_sub & test_sub:
                raise RuntimeError(f"Subject leakage in repeat {repeat}, outer fold {fold}")
            for idx in tr:
                all_split_rows.append({
                    "record_id": meta.iloc[idx]["record_id"], "id_subject": int(groups.iloc[idx]),
                    "repeat": repeat, "fold": fold, "role": "train"
                })
            for idx in te:
                all_split_rows.append({
                    "record_id": meta.iloc[idx]["record_id"], "id_subject": int(groups.iloc[idx]),
                    "repeat": repeat, "fold": fold, "role": "test"
                })

        save_table(pd.DataFrame(all_split_rows), run_dir / "splits.parquet")

        for modality, X in Xs.items():
            for model_name in cfg["models"]["enabled"]:
                for fold, (tr, te) in enumerate(split_list, 1):
                    unit_no += 1
                    stem = f"r{repeat:02d}_{model_name}_{modality}_f{fold:02d}"

                    if is_resume and _unit_is_complete(run_dir, stem):
                        pr = _load_table(run_dir / "fold_outputs" / f"pred_{stem}.parquet")
                        fm = _load_json(run_dir / "fold_outputs" / f"metric_{stem}.json")
                        bp = _load_json(run_dir / "fold_outputs" / f"params_{stem}.json")
                        pred_frames.append(pr); fold_metrics.append(fm); best_params.append(bp)
                        skipped += 1
                        reporter.warning(
                            f"Resume skip [{unit_no}/{total_units}] already complete: "
                            f"repeat={repeat} model={model_name.upper()} modality={modality} outer={fold}/{outer_n}"
                        )
                        continue

                    t0 = time.time()
                    Xtr, Xte = X.iloc[tr], X.iloc[te]
                    ytr, yte = y.iloc[tr], y.iloc[te]
                    gtr = groups.iloc[tr]
                    n_train_sub = gtr.nunique(); n_test_sub = groups.iloc[te].nunique()

                    inner = StratifiedGroupKFold(
                        n_splits=inner_n,
                        shuffle=True,
                        random_state=repeat_seed + fold,
                    )
                    inner_splits = list(inner.split(Xtr, ytr, gtr))
                    for itr, ite in inner_splits:
                        if set(gtr.iloc[itr]) & set(gtr.iloc[ite]):
                            raise RuntimeError(f"Subject leakage in inner CV repeat={repeat}, outer={fold}")

                    pipe, grid = build_model(model_name, cfg, repeat_seed + fold)
                    n_candidates = _candidate_count(grid)
                    n_fits = n_candidates * len(inner_splits)
                    reporter.step(
                        f"[{unit_no}/{total_units}] repeat={repeat}/{repeats} | model={model_name.upper()} | "
                        f"modality={modality} | outer={fold}/{outer_n} | train_sub={n_train_sub} | test_sub={n_test_sub}"
                    )
                    reporter.info(f"Grid search: {n_candidates} candidates x {len(inner_splits)} inner folds = {n_fits} fits")

                    search = GridSearchCV(
                        pipe, grid, scoring=scoring, cv=inner_splits, n_jobs=n_jobs, refit=True,
                        return_train_score=False, error_score="raise"
                    )

                    # Suppress only the known threadpoolctl OpenMP warning. ConvergenceWarning
                    # remains visible/captured because it is scientifically useful.
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", ConvergenceWarning)
                        warnings.filterwarnings(
                            "ignore",
                            message=r".*Found Intel OpenMP.*",
                            category=RuntimeWarning,
                        )
                        with threadpool_limits(limits=1):
                            search.fit(Xtr, ytr)

                    convergence_messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                    if convergence_messages:
                        reporter.warning(
                            f"{model_name.upper()} {modality} r{repeat} f{fold}: "
                            f"{len(convergence_messages)} convergence warning(s) during fit/refit"
                        )

                    reporter.info(
                        f"Best params {model_name.upper()} {modality} r{repeat} f{fold}: "
                        f"{json.dumps(search.best_params_, sort_keys=True, default=str)} | "
                        f"best_inner_{scoring}={search.best_score_:.4f}"
                    )

                    pred = search.predict(Xte)
                    score, score_type = continuous_score(search.best_estimator_, Xte)
                    fm = compute_metrics(yte, pred, score)
                    cw = selected_class_weight(search.best_params_)
                    diag = estimator_diagnostics(search.best_estimator_)
                    elapsed = time.time() - t0
                    fm.update({
                        "run_id": run_id, "repeat": repeat, "modality": modality, "model": model_name,
                        "fold": fold, "fit_seconds": elapsed, "best_inner_score": search.best_score_,
                        "selected_class_weight": cw,
                        "n_iter": diag.get("n_iter"),
                        "max_iter": diag.get("max_iter"),
                        "reached_max_iter": diag.get("reached_max_iter"),
                        "best_validation_score": diag.get("best_validation_score"),
                        "convergence_warning_count": len(convergence_messages),
                    })
                    fold_metrics.append(fm)

                    bp = {
                        "run_id": run_id, "repeat": repeat, "modality": modality, "model": model_name,
                        "fold": fold, "best_inner_score": search.best_score_, "selected_class_weight": cw,
                        "best_params": json.dumps(search.best_params_, sort_keys=True),
                        "n_iter": diag.get("n_iter"),
                        "max_iter": diag.get("max_iter"),
                        "reached_max_iter": diag.get("reached_max_iter"),
                        "best_validation_score": diag.get("best_validation_score"),
                        "convergence_warning_count": len(convergence_messages),
                        "convergence_warnings": json.dumps(convergence_messages, ensure_ascii=False),
                    }
                    best_params.append(bp)

                    pr = meta.iloc[te].copy()
                    pr["repeat"] = repeat; pr["fold"] = fold
                    pr["modality"] = modality; pr["model"] = model_name
                    pr["selected_class_weight"] = cw
                    pr["y_true"] = yte.to_numpy(); pr["y_pred"] = pred
                    pr["score_error"] = score if score is not None else np.nan
                    pr["score_type"] = score_type
                    pred_frames.append(pr)

                    save_table(pr, run_dir / "fold_outputs" / f"pred_{stem}.parquet")
                    save_json(fm, run_dir / "fold_outputs" / f"metric_{stem}.json")
                    save_json(bp, run_dir / "fold_outputs" / f"params_{stem}.json")

                    if cfg["output"].get("save_inner_cv", True):
                        cvdf = pd.DataFrame(search.cv_results_)
                        save_table(cvdf, run_dir / "inner_cv" / f"{stem}.parquet")
                    if cfg["compute"].get("save_models", True):
                        joblib.dump(search.best_estimator_, run_dir / "models" / f"{stem}.joblib", compress=3)

                    pd.DataFrame(fold_metrics).to_csv(run_dir / "metrics_folds.csv", index=False)
                    pd.DataFrame(best_params).to_csv(run_dir / "best_params.csv", index=False)
                    iter_text = ""
                    if diag.get("n_iter") is not None:
                        estimator_class = diag.get("estimator_class")
                        max_iter = diag.get("max_iter")
                        if estimator_class == "SVC":
                            limit_text = "unlimited" if max_iter == -1 else str(max_iter)
                            iter_text = f" | solver_iter={diag.get('n_iter')} | max_iter={limit_text}"
                        else:
                            conv = "yes" if diag.get("reached_max_iter") is False else ("no/limit" if diag.get("reached_max_iter") else "unknown")
                            iter_text = f" | iter={diag.get('n_iter')}/{max_iter} | converged={conv}"
                    reporter.success(
                        f"Finished {model_name.upper()} {modality} r{repeat} f{fold} in {elapsed:.1f}s | "
                        f"inner={search.best_score_:.3f} | class_weight={cw}{iter_text} | {_short_metric(fm)}"
                    )

    if not pred_frames:
        raise RuntimeError("No completed fold outputs were available.")

    preds = pd.concat(pred_frames, ignore_index=True)
    save_table(preds, run_dir / "predictions.parquet")
    pd.DataFrame(fold_metrics).to_csv(run_dir / "metrics_folds.csv", index=False)
    pd.DataFrame(best_params).to_csv(run_dir / "best_params.csv", index=False)

    oof = []
    for (repeat, modality, model), g in preds.groupby(["repeat", "modality", "model"], sort=True):
        if g["record_id"].duplicated().any():
            raise RuntimeError(f"Duplicate OOF record in repeat={repeat}, {modality}/{model}")
        m = compute_metrics(g["y_true"], g["y_pred"], g["score_error"])
        m.update({"run_id": run_id, "repeat": repeat, "modality": modality, "model": model})
        oof.append(m)
    oof_df = pd.DataFrame(oof)
    oof_df.to_csv(run_dir / "metrics_oof.csv", index=False)

    metric_cols = [
        "accuracy", "balanced_accuracy", "f1_error", "precision_error", "sensitivity",
        "specificity", "mcc", "fpr", "roc_auc", "average_precision", "predicted_error_rate"
    ]
    summary_rows = []
    for (modality, model), g in oof_df.groupby(["modality", "model"], sort=True):
        row = {"run_id": run_id, "modality": modality, "model": model, "n_repeats": len(g)}
        for c in metric_cols:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_sd"] = g[c].std(ddof=1) if len(g) > 1 else np.nan
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics_oof_summary.csv", index=False)

    if skipped:
        reporter.info(f"Resume reused {skipped}/{total_units} completed outer-fit units.")
    reporter.success(f"Completed run in {(time.time() - run_t0) / 60:.1f} min")
    reporter.success(f"Run directory: {run_dir}")
    return run_dir
