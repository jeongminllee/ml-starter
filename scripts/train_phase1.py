from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- W&B 모드 자동결정 : 기본 online, 키 없으면 offline ---
if os.getenv("WANDB_MODE") is None:
    if os.getenv("WANDB_API_KEY") or (Path.home() / ".netrc").exists():
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"

import wandb

PROJECT = "ml-starter"
OUT_DIR = Path("outputs/phase1")
DATA_DIR = Path(".data")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_titanic() -> tuple[pd.DataFrame, pd.Series]:
    """캐시된 CSV가 있으면 사용, 없으면 OpenML에서 가져와 저장."""
    cache = DATA_DIR / "titanic.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        # 네트워크 필요. 실패하면 사용자에게 캐시 파일을 두라고 안내
        df = fetch_openml("titanic", version=1, as_frame=True).frame
        df.to_csv(cache, index=False)

    # 타깃/피처 선정(심플 버전)
    y = df["survived"].astype(int)
    X = df.drop(columns=["survived"])

    # 자주 쓰는 열만
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    X = X[keep]

    # dtype 정리
    num_cols = ["age", "sibsp", "parch", "fare", "pclass"]
    cat_cols = ["sex", "embarked"]
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in cat_cols:
        X[c] = X[c].astype("category")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()

    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    # cat_pipe = Pipeline(
    #     [
    #         ("imputer", SimpleImputer(strategy="most_frequent")),
    #         ("ohe", OneHotEncoder(handle_unknown="ignore")),
    #     ]
    # )
    try:
        # scikit-learn >= 1.2
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    except TypeError:
        # scikit-learn < 1.2
        cat_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

    pre = ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    return pre


def plot_and_save_cm(y_true, y_pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def run_baseline():
    X, y = load_titanic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pre = build_preprocessor(X)

    models = {
        "logreg": Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]),
        "rf": Pipeline(
            [("pre", pre), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))]
        ),
    }

    scores = {}

    for name, pipe in models.items():
        _ = wandb.init(
            project=PROJECT,
            name=f"p1-baseline-{name}",
            config={"model": name},
            settings=wandb.Settings(save_code=False, code_dir="."),
            reinit="finish_previous",
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)
        scores[name] = {"acc": acc, "f1": f1}

        # 로그 & 이미지 저장
        png = OUT_DIR / f"{name}_cm.png"
        plot_and_save_cm(y_test, pred, png, f"Confusion Matrix - {name}")
        wandb.log({"accuracy": acc, "f1": f1})
        wandb.save(str(png))
        wandb.finish()

    print("=== Summary (baseline) ===")
    print(scores)
    return X_train, X_test, y_train, y_test, pre, scores


def run_tuning(X_train, X_test, y_train, y_test, pre):
    # 로지스틱 간단 튜닝
    grid = GridSearchCV(
        estimator=Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))]),
        param_grid={
            "clf__C": [0.1, 0.3, 1.0, 3.0],
            "clf__class_weight": [None, "balanced"],
        },
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
    )

    _ = wandb.init(
        project=PROJECT,
        name="p1-tuned-logreg",
        config={"model": "logreg", "search": "grid"},
        settings=wandb.Settings(save_code=False, code_dir="."),
        reinit="finish_previous",
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    pred = best.predict(X_test)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)
    wandb.log(
        {"tuned/logreg/accuracy": acc, "tuned/logreg/f1": f1, "best_params": grid.best_params_}
    )

    # Permutation importance (상위 10개 표시)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(best, X_test, y_test, scoring="f1", n_repeats=8, random_state=42)
    # feat = best.named_steps["pre"].get_feature_names_out()    # 수정 전
    feat = X_test.columns.to_numpy()
    imp = (
        pd.DataFrame(
            {"feature": feat, "importance": r.importances_mean}  # 수정 전 : np.array(feat)
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )
    wandb.log({"tuned/logreg/top10_importance": wandb.Table(dataframe=imp)})

    png = OUT_DIR / "logreg_tuned_cm.png"
    plot_and_save_cm(y_test, pred, png, "Confusion Matrix - logreg (tuned)")
    wandb.save(str(png))
    wandb.finish()

    print("=== Summary (tuned logreg) ===")
    print({"acc": acc, "f1": f1, "best_params": grid.best_params_})


def run_rf_tuning(X_train, X_test, y_train, y_test, pre):
    """Day2: RandomForest 소규모 그리드 서치 + W&B 로깅"""
    rf_pipe = Pipeline(
        [
            ("pre", pre),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 6, 12],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__criterion": ["gini", "entropy"],
        # "clf__class_weight" : [None, "balanced"],
    }

    grid = GridSearchCV(
        estimator=rf_pipe,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    _ = wandb.init(
        project=PROJECT,
        name="p1-tuned-rf",
        config={
            "model": "rf",
            "search": "grid",
        },
        settings=wandb.Settings(save_code=False, code_dir="."),
        reinit="finish_previous",
    )

    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    pred = best.predict(X_test)
    acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)
    wandb.log(
        {
            "tuned/rf/accuracy": acc,
            "tuned/rf/f1": f1,
            "tuned/rf/best_params": grid.best_params_,
        }
    )

    # Permutation importance (원본 피처 기준: 길이 불일치 방지)
    r = permutation_importance(best, X_test, y_test, scoring="f1", n_repeats=8, random_state=42)
    feat = X_test.columns.to_numpy()
    imp = (
        pd.DataFrame({"feature": feat, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    wandb.log({"tuned/rf/top10_importance": wandb.Table(dataframe=imp)})

    # 혼동행렬 저장/로그
    png = OUT_DIR / "rf_tuned_cm.png"
    plot_and_save_cm(y_test, pred, png, "Confusion Matrix - RF (tuned)")
    wandb.save(str(png))
    wandb.finish()

    print("=== Summary (tuned RF) ===")
    print({"acc": acc, "f1": f1, "best_params": grid.best_params_})


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pre, _ = run_baseline()
    run_tuning(X_train, X_test, y_train, y_test, pre)
    run_rf_tuning(X_train, X_test, y_train, y_test, pre)
