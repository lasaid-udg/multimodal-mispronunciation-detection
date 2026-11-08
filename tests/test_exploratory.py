"""
Tests for auxiliary exploratory helpers (moved to md_emg_extras in Prompt 3).

These functions are NOT part of the core reproducible pipeline; they live in
src/md_emg_extras/exploratory.py and require fitted outer-fold .joblib models
that are only available after a full run with --save-models.
"""
from md_emg_extras.exploratory import emg_feature_family, feature_family


def test_emg_feature_families():
    assert emg_feature_family('willison_amplitude_CH1') == 'EMG time-domain'
    assert emg_feature_family('median_frequency_CH2') == 'EMG frequency-domain'
    assert emg_feature_family('CH3_energy_point_7') == 'EMG STFT'
    assert emg_feature_family('CH4_mfcc_13') == 'EMG MFCC'


def test_acoustic_family_wins():
    assert feature_family('duration', {'duration'}) == 'Acoustic'

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from md_emg_extras.exploratory import _grouped_permutation_importance


def test_grouped_permutation_importance_shape():
    X = pd.DataFrame({
        'a1': [0, 0, 1, 1, 0, 1],
        'a2': [1, 1, 0, 0, 1, 0],
        'e1': [0, 1, 0, 1, 0, 1],
    })
    y = pd.Series([0, 0, 1, 1, 0, 1])
    clf = DummyClassifier(strategy='most_frequent').fit(X, y)
    rows = _grouped_permutation_importance(
        clf, X, y, {'Acoustic': ['a1', 'a2'], 'EMG time-domain': ['e1']},
        n_repeats=2, seed=1,
    )
    assert len(rows) == 2
    assert {r['family'] for r in rows} == {'Acoustic', 'EMG time-domain'}
    assert {r['n_features'] for r in rows} == {1, 2}
