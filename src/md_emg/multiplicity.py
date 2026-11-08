from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

PHONEME_LABELS = {'t': '/t/', 'd': '/d/', 'theta': '/θ/', 'eth': '/ð/'}


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float)
    if np.any(~np.isfinite(p)):
        raise ValueError('FDR input contains non-finite p-values.')
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * m / np.arange(1, m + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q


def parse_run_name(run_name: str) -> dict:
    name = run_name.lower()
    m = re.match(r'general_(ch[1-4])_', name)
    if m:
        ch = m.group(1).upper()
        return {'family': 'general', 'scenario': ch, 'phoneme': 'all', 'channel': ch}
    m = re.match(r'ph_(t|d|theta|eth)_(ch[1-4])_', name)
    if m:
        ph = PHONEME_LABELS[m.group(1)]
        ch = m.group(2).upper()
        return {'family': 'phoneme_channel', 'scenario': f'{ph} {ch}', 'phoneme': ph, 'channel': ch}
    raise ValueError(f'Could not parse scenario from run directory: {run_name}')


def apply_fdr(root: str | Path, output: str | Path | None = None, alpha: float = 0.05) -> Path:
    root = Path(root).resolve()
    rows = []
    files = sorted(root.glob('*/analysis/paired_deltas.csv'))
    if not files:
        raise FileNotFoundError(f'No */analysis/paired_deltas.csv files found under {root}')
    for f in files:
        info = parse_run_name(f.parent.parent.name)
        df = pd.read_csv(f)
        ba = df.loc[df['metric'] == 'balanced_accuracy']
        if len(ba) != 7:
            raise ValueError(f'{f}: expected 7 balanced-accuracy rows, found {len(ba)}')
        for _, r in ba.iterrows():
            rows.append({
                **info,
                'model': r['model'],
                'delta_ba': r['delta_estimate'],
                'ci_low': r['delta_ci95_low'],
                'ci_high': r['delta_ci95_high'],
                'p_raw': r['permutation_p'],
            })
    result = pd.DataFrame(rows)
    counts = result['family'].value_counts().to_dict()
    if counts.get('general', 0) != 28:
        raise ValueError(f"Expected 28 general comparisons; found {counts.get('general', 0)}")
    if counts.get('phoneme_channel', 0) != 112:
        raise ValueError(f"Expected 112 phoneme-level comparisons; found {counts.get('phoneme_channel', 0)}")
    result['q_fdr'] = np.nan
    for family, g in result.groupby('family'):
        result.loc[g.index, 'q_fdr'] = benjamini_hochberg(g['p_raw'].to_numpy())
    result['fdr_sig'] = result['q_fdr'] < float(alpha)
    result = result.sort_values(['family', 'scenario', 'channel', 'model']).reset_index(drop=True)
    output = Path(output).resolve() if output is not None else root / 'FDR_balanced_accuracy_all_models.csv'
    result.to_csv(output, index=False)
    return output
