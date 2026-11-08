from pathlib import Path

import pandas as pd
import pytest

from md_emg.features import build_feature_registry, predictor_columns


def test_real_feature_counts():
    path = Path("data/raw/Nuevo_final_df.csv")
    if not path.exists():
        pytest.skip("Real dataset not present in this checkout")
    df=pd.read_csv(path,nrows=1)
    r=build_feature_registry(df.columns.tolist())
    assert len(r["audio"])==84
    for ch in ["ch1","ch2","ch3","ch4"]:
        assert len(r[ch])==101
    a,e=predictor_columns(r,"ch2",True)
    assert len(a)==84 and len(e)==99
