from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import wandb
from ml_starter.data import load_titanic
from ml_starter.pipeline import build__preprocessor, build_models

if os.getenv("WANDB_MODE") is None:
    if os.getenv("WANDB_API_KEY") or (Path.home() / ".netrc").exists():
        os.environ["WANDB_MODE"] = "online"
    else:
        os.environ["WANDB_MODE"] = "offline"


def plot_cm(name, y_true, y_pred, out_dir: Path) -> Path:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = disp.figure_
    fig.suptitle(f"{name} - confusion matrix")
    out = out_dir / f"{name}_cm.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    X, y = load_titanic()
    pre = build__preprocessor(X)
    models = build_models(pre)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = {}

    for name, pipe in models.items():
        _ = wandb.init(
            project="ml-starter",
            name=f"phase1-{name}",
            config={"model": name},
            reinit="finish_previous",
        )

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)
        cm_path = plot_cm(name, y_test, pred, Path("outputs/phase1"))
        wandb.log({"accuracy": acc, "f1": f1, "confusion_matrix": wandb.Image(str(cm_path))})
        wandb.finish()
        results[name] = {"acc": acc, "f1": f1}

    print("=== Summary ===")
    print(results)


if __name__ == "__main__":
    main()
