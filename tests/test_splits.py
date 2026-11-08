from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import StratifiedGroupKFold


def test_no_subject_overlap():
    path = Path("data/raw/Nuevo_final_df.csv")
    if not path.exists():
        pytest.skip("Real dataset not present in this checkout")
    df=pd.read_csv(path)
    y=df["Clase"].astype(int); g=df["id_subject"].astype(int)
    X=df[["duration"]]
    cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
    for tr,te in cv.split(X,y,g):
        assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te]))
