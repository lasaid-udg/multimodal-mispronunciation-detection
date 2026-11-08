import numpy as np
import pandas as pd

from md_emg.analysis import analyze_run


def test_analyze_run_cluster_bootstrap(tmp_path):
    rows = []
    for repeat in [1, 2]:
        for sid in [1, 2, 3, 4]:
            y = sid % 2
            for modality in ["A", "A_CH2"]:
                pred = y if modality == "A_CH2" else (y if sid != 2 else 0)
                rows.append({
                    "id_subject": sid,
                    "record_id": f"{sid}_1",
                    "repeat": repeat,
                    "fold": 1,
                    "modality": modality,
                    "model": "svm",
                    "y_true": y,
                    "y_pred": pred,
                    "score_error": 0.9 if pred == 1 else 0.1,
                })
    pd.DataFrame(rows).to_csv(tmp_path / "predictions.csv", index=False)
    out = analyze_run(tmp_path, bootstrap=20, permutations=20, seed=1, metrics=["balanced_accuracy"])
    assert (out / "repeat_metrics.csv").exists()
    assert (out / "bootstrap_ci.csv").exists()
    d = pd.read_csv(out / "paired_deltas.csv")
    assert d.shape[0] == 1
    assert d.loc[0, "comparison"] == "A_CH2 - A"
    assert d.loc[0, "delta_estimate"] >= 0
