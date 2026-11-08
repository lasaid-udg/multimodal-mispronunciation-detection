from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from md_emg.data import build_scenario, read_csv


def _load_table(base: Path) -> pd.DataFrame:
    if base.exists():
        return pd.read_parquet(base)
    csv = base.with_suffix('.csv')
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(base)


def _load_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def emg_feature_family(name: str) -> str:
    """Map one EMG predictor to the feature families used in the manuscript."""
    n = name.lower()
    if any(k in n for k in ('willison_amplitude', 'zero_crossings', 'mean_emg', 'mean_absolute_value')):
        return 'EMG time-domain'
    if any(k in n for k in ('median_frequency', 'mean_frequency', 'spectral_moment_')):
        return 'EMG frequency-domain'
    if any(k in n for k in ('peak_freq_point_', 'energy_point_', 'freq_var_point_', 'centroid_point_')):
        return 'EMG STFT'
    if '_mfcc_' in n:
        return 'EMG MFCC'
    return 'EMG other'


def feature_family(name: str, acoustic_features: set[str]) -> str:
    return 'Acoustic' if name in acoustic_features else emg_feature_family(name)




def _grouped_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    families: dict[str, list[str]],
    *,
    n_repeats: int,
    seed: int,
) -> list[dict]:
    """Jointly permute all columns in each feature family on one held-out set.

    A single row permutation is applied to every column in a family so the
    within-family multivariate structure is retained while its association with
    the target is broken. Importance is the decrease in balanced accuracy from
    the unpermuted held-out baseline.
    """
    baseline = balanced_accuracy_score(y, estimator.predict(X))
    rows: list[dict] = []
    for family_idx, (family, columns) in enumerate(families.items()):
        rng = np.random.default_rng(int(seed) + 10000 * (family_idx + 1))
        drops = []
        for _ in range(int(n_repeats)):
            order = rng.permutation(len(X))
            Xp = X.copy()
            # Use one common permutation for the whole family. This preserves
            # correlations among features in the group while disconnecting the
            # group from y.
            Xp.loc[:, columns] = X.iloc[order][columns].to_numpy()
            permuted = balanced_accuracy_score(y, estimator.predict(Xp))
            drops.append(float(baseline - permuted))
        rows.append({
            'family': family,
            'n_features': int(len(columns)),
            'baseline_balanced_accuracy': float(baseline),
            'importance_mean': float(np.mean(drops)),
            'importance_sd_within_permutation': float(np.std(drops, ddof=0)),
        })
    return rows

def _data_path_for_run(run_dir: Path, cfg: dict, data_path: str | Path | None) -> Path:
    if data_path is not None:
        return Path(data_path).resolve()
    manifest = run_dir / 'manifest.json'
    if manifest.exists():
        p = Path(_load_json(manifest).get('data_path', ''))
        if p.exists():
            return p.resolve()
    p = Path(cfg['data']['path'])
    if p.is_absolute() and p.exists():
        return p.resolve()

    # When a completed run is copied back into the project (for example
    # <project>/runs_final_r10/<run>), resolve the original relative data path
    # from the project root so --data is not needed.
    project_candidate = run_dir.parent.parent / p
    if project_candidate.exists():
        return project_candidate.resolve()

    # Also allow execution from the project root as a convenience.
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate.resolve()

    raise FileNotFoundError(
        'Could not resolve the dataset used by this run. Expected it at the original path, '
        f'at {project_candidate}, or relative to the current directory. Pass it explicitly with --data.'
    )


def _best_multimodal_model(run_dir: Path) -> tuple[str, str]:
    summary = run_dir / 'metrics_oof_summary.csv'
    if summary.exists():
        s = pd.read_csv(summary)
    else:
        repeat_summary = run_dir / 'analysis' / 'repeat_summary.csv'
        if not repeat_summary.exists():
            raise FileNotFoundError('metrics_oof_summary.csv or analysis/repeat_summary.csv is required.')
        s = pd.read_csv(repeat_summary)
    multimodal = s.loc[s['modality'] != 'A'].copy()
    if multimodal.empty:
        raise ValueError('No A+EMG modality was found in the run summary.')
    row = multimodal.sort_values('balanced_accuracy_mean', ascending=False).iloc[0]
    return str(row['model']), str(row['modality'])


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically so an interruption does not leave a half-written file."""
    tmp = path.parent / '.pi_tmp.csv'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f'{int(minutes)}m {sec:.0f}s'
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours)}h {int(minutes)}m'


def run_permutation_importance(
    run_dir: str | Path,
    *,
    data_path: str | Path | None = None,
    n_repeats: int = 10,
    seed: int = 2026,
    n_jobs: int = 1,
    resume: bool = True,
) -> Path:
    """Permutation importance on held-out outer-test folds for the best A+EMG model.

    The model is chosen descriptively by its mean OOF balanced accuracy within the
    completed scenario. Each already-fitted outer-fold estimator is evaluated only
    on its corresponding held-out participant-grouped test fold. No refitting is done.

    Progress is printed fold by fold. Completed folds are saved incrementally and can
    be resumed safely if the process is interrupted.
    """
    run_dir = Path(run_dir).resolve()
    cfg = _load_yaml(run_dir / 'config.yaml')
    p = _data_path_for_run(run_dir, cfg, data_path)
    df = read_csv(p)
    Xs, y, groups, meta, feature_info = build_scenario(df, cfg)

    model, modality = _best_multimodal_model(run_dir)
    if modality not in Xs:
        raise ValueError(f'Multimodal modality {modality} is not available after rebuilding the scenario.')
    X = Xs[modality].copy()
    acoustic = set(feature_info['audio'])

    pred = _load_table(run_dir / 'predictions.parquet')
    pred = pred.loc[(pred['model'] == model) & (pred['modality'] == modality)].copy()
    if pred.empty:
        raise ValueError(f'No predictions found for {model}/{modality}.')

    meta2 = meta.reset_index(drop=True).copy()
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    record_to_pos = {rid: i for i, rid in enumerate(meta2['record_id'].tolist())}

    out_dir = run_dir / 'exploratory'
    out_dir.mkdir(exist_ok=True)
    fold_path = out_dir / 'permutation_importance_outer_folds.csv'
    grouped_fold_path = out_dir / 'permutation_importance_grouped_outer_folds.csv'
    state_path = out_dir / 'permutation_importance_state.json'
    manifest_path = out_dir / 'permutation_importance_manifest.json'

    expected_state = {
        'analysis_version': '0.3.5.1-grouped',
        'model': model,
        'modality': modality,
        'scoring': 'balanced_accuracy',
        'permutation_repeats_per_outer_test': int(n_repeats),
        'seed': int(seed),
        'n_features': int(X.shape[1]),
        'data_path': str(p),
    }

    if not resume:
        for q in [fold_path, state_path, manifest_path,
                  out_dir / 'pi_features.csv',
                  out_dir / 'pi_families_individual.csv',
                  grouped_fold_path,
                  out_dir / 'pi_grouped_families.csv']:
            if q.exists():
                q.unlink()

    if resume and state_path.exists():
        old_state = _load_json(state_path)
        comparable = {k: old_state.get(k) for k in expected_state}
        # v0.3.5 checkpoints are scientifically identical and can be resumed safely.
        if comparable.get('analysis_version') == '0.3.5-grouped':
            comparable['analysis_version'] = '0.3.5.1-grouped'
        if comparable != expected_state:
            raise ValueError(
                'Existing permutation-importance partial results were created with different settings. '
                'Use --restart to discard them and start again.\n'
                f'Existing: {comparable}\nCurrent: {expected_state}'
            )
    else:
        with state_path.open('w', encoding='utf-8') as f:
            json.dump({**expected_state, 'status': 'running'}, f, indent=2, ensure_ascii=False)

    if resume and fold_path.exists():
        fold_df = pd.read_csv(fold_path)
        rows = fold_df.to_dict('records')
    else:
        rows = []
    if resume and grouped_fold_path.exists():
        grouped_rows = pd.read_csv(grouped_fold_path).to_dict('records')
    else:
        grouped_rows = []

    individual_completed = {
        (int(r), int(f))
        for r, f in pd.DataFrame(rows)[['repeat', 'fold']].drop_duplicates().itertuples(index=False, name=None)
    } if rows else set()
    grouped_completed = {
        (int(r), int(f))
        for r, f in pd.DataFrame(grouped_rows)[['repeat', 'fold']].drop_duplicates().itertuples(index=False, name=None)
    } if grouped_rows else set()
    completed = individual_completed & grouped_completed

    family_columns: dict[str, list[str]] = {}
    for col in X.columns:
        family_columns.setdefault(feature_family(col, acoustic), []).append(col)

    fold_groups = list(pred.groupby(['repeat', 'fold'], sort=True))
    total_folds = len(fold_groups)
    start_time = time.perf_counter()
    newly_done = 0

    scenario = cfg.get('scenario', {})
    print('')
    print('Permutation importance')
    print(f"  Scenario: phoneme={scenario.get('phoneme')} | channel={scenario.get('channel')}")
    print(f'  Selected A+EMG model: {model} ({modality})')
    print(f'  Features: {X.shape[1]} | permutation repeats/feature or family: {int(n_repeats)}')
    print(f'  Grouped families: {len(family_columns)} ({", ".join(family_columns)})')
    print(f'  Outer test folds: {total_folds} | n_jobs: {int(n_jobs)}')
    print(f'  Output: {out_dir}')
    if completed:
        print(f'  Resume: {len(completed)}/{total_folds} folds already completed')
    print('', flush=True)

    for ordinal, ((repeat, fold), g) in enumerate(fold_groups, start=1):
        key = (int(repeat), int(fold))
        if key in completed:
            print(f'[{ordinal}/{total_folds}] Repeat {int(repeat)}/{cfg["cv"]["repeats"]}, '
                  f'fold {int(fold)}/{cfg["cv"]["outer_folds"]}: individual + grouped already completed, skipping.', flush=True)
            continue

        positions = [record_to_pos[r] for r in g['record_id'].tolist()]
        Xte = X.iloc[positions]
        yte = y.iloc[positions]
        model_file = run_dir / 'models' / f'r{int(repeat):02d}_{model}_{modality}_f{int(fold):02d}.joblib'
        if not model_file.exists():
            raise FileNotFoundError(
                f'{model_file} was not found. Permutation importance requires the fitted outer-fold models. '
                'Run experiments with compute.save_models=true and without --no-save-models.'
            )

        print(f'[{ordinal}/{total_folds}] Repeat {int(repeat)}/{cfg["cv"]["repeats"]}, '
              f'fold {int(fold)}/{cfg["cv"]["outer_folds"]}: starting '
              f'({len(Xte)} test records)...', flush=True)
        fold_start = time.perf_counter()
        estimator = joblib.load(model_file)

        if key not in individual_completed:
            print('    individual features...', flush=True)
            pi = permutation_importance(
                estimator, Xte, yte,
                scoring='balanced_accuracy',
                n_repeats=int(n_repeats),
                random_state=int(seed + int(repeat) * 100 + int(fold)),
                n_jobs=int(n_jobs),
            )
            fold_rows = []
            for feature, mean_imp, sd_imp in zip(X.columns, pi.importances_mean, pi.importances_std):
                fold_rows.append({
                    'repeat': int(repeat), 'fold': int(fold), 'model': model, 'modality': modality,
                    'feature': feature, 'family': feature_family(feature, acoustic),
                    'importance_mean': float(mean_imp), 'importance_sd_within_permutation': float(sd_imp),
                })
            rows.extend(fold_rows)
            _atomic_write_csv(pd.DataFrame(rows), fold_path)
            individual_completed.add(key)
        else:
            print('    individual features: checkpoint found, skipping.', flush=True)

        if key not in grouped_completed:
            print('    grouped families...', flush=True)
            family_rows = _grouped_permutation_importance(
                estimator, Xte, yte, family_columns,
                n_repeats=int(n_repeats),
                seed=int(seed + int(repeat) * 100 + int(fold)),
            )
            for r in family_rows:
                r.update({
                    'repeat': int(repeat), 'fold': int(fold),
                    'model': model, 'modality': modality,
                })
            grouped_rows.extend(family_rows)
            _atomic_write_csv(pd.DataFrame(grouped_rows), grouped_fold_path)
            grouped_completed.add(key)
        else:
            print('    grouped families: checkpoint found, skipping.', flush=True)

        completed = individual_completed & grouped_completed
        newly_done += 1

        elapsed = time.perf_counter() - start_time
        fold_elapsed = time.perf_counter() - fold_start
        remaining = total_folds - len(completed)
        eta = (elapsed / newly_done) * remaining if newly_done > 0 else 0.0
        print(f'    done in {_format_duration(fold_elapsed)} | '
              f'overall {len(completed)}/{total_folds} ({100*len(completed)/total_folds:.1f}%) | '
              f'elapsed {_format_duration(elapsed)} | ETA ~{_format_duration(eta)}', flush=True)

        with state_path.open('w', encoding='utf-8') as f:
            json.dump({**expected_state, 'status': 'running',
                       'completed_outer_tests': len(completed), 'total_outer_tests': total_folds},
                      f, indent=2, ensure_ascii=False)

    fold_df = pd.DataFrame(rows)
    if fold_df.empty:
        raise RuntimeError('No permutation-importance results were produced.')

    feature_summary = (
        fold_df.groupby(['family', 'feature'], as_index=False)
        .agg(
            importance_mean=('importance_mean', 'mean'),
            importance_sd_across_outer_tests=('importance_mean', 'std'),
            n_outer_tests=('importance_mean', 'size'),
        )
        .sort_values('importance_mean', ascending=False)
    )
    _atomic_write_csv(feature_summary, out_dir / 'pi_features.csv')

    family_summary = (
        feature_summary.groupby('family', as_index=False)
        .agg(
            n_features=('feature', 'size'),
            mean_importance_per_feature=('importance_mean', 'mean'),
            total_importance=('importance_mean', 'sum'),
            positive_importance_sum=('importance_mean', lambda s: float(s.clip(lower=0).sum())),
        )
        .sort_values('mean_importance_per_feature', ascending=False)
    )
    _atomic_write_csv(family_summary, out_dir / 'pi_families_individual.csv')

    grouped_df = pd.DataFrame(grouped_rows)
    if grouped_df.empty:
        raise RuntimeError('No grouped permutation-importance results were produced.')
    grouped_summary = (
        grouped_df.groupby(['family', 'n_features'], as_index=False)
        .agg(
            importance_mean=('importance_mean', 'mean'),
            importance_sd_across_outer_tests=('importance_mean', 'std'),
            n_outer_tests=('importance_mean', 'size'),
            mean_baseline_balanced_accuracy=('baseline_balanced_accuracy', 'mean'),
        )
        .sort_values('importance_mean', ascending=False)
    )
    _atomic_write_csv(grouped_summary, out_dir / 'pi_grouped_families.csv')

    meta_out = {
        'analysis': 'exploratory individual and grouped permutation importance',
        'analysis_version': '0.3.5.1-grouped',
        'model_selection': 'best A+EMG model by mean OOF balanced accuracy within this scenario',
        'model': model,
        'modality': modality,
        'scoring': 'balanced_accuracy',
        'permutation_repeats_per_outer_test': int(n_repeats),
        'n_outer_tests': int(fold_df[['repeat', 'fold']].drop_duplicates().shape[0]),
        'data_path': str(p),
        'resumable': True,
        'grouped_definition': 'For grouped importance, all columns in a feature family are permuted jointly using one common row permutation, preserving within-family structure while breaking association with the target.',
        'families': {k: len(v) for k, v in family_columns.items()},
        'note': 'Both individual and grouped importance were evaluated only on held-out outer-test folds using already-fitted nested-CV estimators; no model refitting or feature selection was performed.',
    }
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)
    with state_path.open('w', encoding='utf-8') as f:
        json.dump({**expected_state, 'status': 'completed',
                   'completed_outer_tests': total_folds, 'total_outer_tests': total_folds},
                  f, indent=2, ensure_ascii=False)

    total_elapsed = time.perf_counter() - start_time
    print('')
    print(f'Permutation importance completed: {total_folds}/{total_folds} outer test folds')
    print(f'Total elapsed this session: {_format_duration(total_elapsed)}')
    print(f'Results: {out_dir}', flush=True)
    return out_dir


def run_pca(
    run_dir: str | Path,
    *,
    data_path: str | Path | None = None,
) -> Path:
    """Descriptive PCA for A and A+EMG within the exact scenario subset.

    PCA is not used for model fitting, feature selection, or inference. Median
    imputation and standardization are applied separately within each modality.
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    cfg = _load_yaml(run_dir / 'config.yaml')
    p = _data_path_for_run(run_dir, cfg, data_path)
    df = read_csv(p)
    Xs, y, groups, meta, feature_info = build_scenario(df, cfg)

    out_dir = run_dir / 'exploratory'
    out_dir.mkdir(exist_ok=True)
    variance = {}
    score_frames = []

    modalities = list(Xs.keys())
    fig, axes = plt.subplots(1, len(modalities), figsize=(6 * len(modalities), 5), squeeze=False)

    for ax, modality in zip(axes[0], modalities):
        X = Xs[modality].reset_index(drop=True)

        # Excluir LPC1-LPC16 solo del PCA exploratorio
        lpc_cols = [f"lpc{i}" for i in range(1, 17)]
        X = X.drop(columns=[c for c in lpc_cols if c in X.columns], errors="ignore")

        prep = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        Z = prep.fit_transform(X)
        pca = PCA(n_components=2)
        scores = pca.fit_transform(Z)
        evr = pca.explained_variance_ratio_
        variance[modality] = [float(evr[0]), float(evr[1])]

        sdf = meta.reset_index(drop=True).copy()
        sdf['class'] = y.reset_index(drop=True).to_numpy()
        sdf['modality'] = modality
        sdf['PC1'] = scores[:, 0]
        sdf['PC2'] = scores[:, 1]
        score_frames.append(sdf)

        for cls, label in [(0, 'Correct'), (1, 'Pronunciation error')]:
            mask = sdf['class'].to_numpy() == cls
            ax.scatter(scores[mask, 0], scores[mask, 1], s=18, alpha=0.65, label=label)
        ax.set_title('Acoustic-only' if modality == 'A' else 'Acoustic + EMG')
        ax.set_xlabel(f'PC1 ({100 * evr[0]:.1f}% variance)')
        ax.set_ylabel(f'PC2 ({100 * evr[1]:.1f}% variance)')
        ax.legend()

    fig.suptitle('Exploratory PCA of channel-specific feature spaces')
    fig.tight_layout()
    fig.savefig(out_dir / 'pca_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    pd.concat(score_frames, ignore_index=True).to_csv(out_dir / 'pca_scores.csv', index=False)
    with (out_dir / 'pca_manifest.json').open('w', encoding='utf-8') as f:
        json.dump({
            'analysis': 'exploratory PCA',
            'modalities': modalities,
            'explained_variance_ratio': variance,
            'data_path': str(p),
            'note': 'PCA was fitted separately for A and A+EMG after median imputation and standardization, solely for descriptive visualization. It was not used in model training, feature selection, or statistical inference.',
        }, f, indent=2, ensure_ascii=False)
    return out_dir


def run_exploratory(
    run_dir: str | Path,
    *,
    data_path: str | Path | None = None,
    permutation_repeats: int = 10,
    seed: int = 2026,
    n_jobs: int = 1,
    resume: bool = True,
) -> Path:
    out = run_permutation_importance(
        run_dir, data_path=data_path, n_repeats=permutation_repeats, seed=seed, n_jobs=n_jobs, resume=resume
    )
    run_pca(run_dir, data_path=data_path)
    return out
