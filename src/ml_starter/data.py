from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml


def load_titanic() -> tuple[pd.DataFrame, pd.Series]:
    """혼합형 컬럼(수치+범주)이 있는 공개 데이터."""
    df = fetch_openml("titanic", version=1, as_frame=True).frame
    y = pd.to_numeric(df["survived"], errors="coerce").astype("Int64").astype(int)
    cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = df[cols].copy()
    for c in ["sex", "embarked"]:
        X[c] = X[c].astype("category")
    return X, y
