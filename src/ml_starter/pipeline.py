from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build__preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    catecorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", numeric, num_cols), ("cat", catecorical, cat_cols)])


def build_models(pre: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]),
        "rf": Pipeline(
            [("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))]
        ),
    }
