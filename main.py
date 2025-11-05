from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import wandb

wandb.init(project="ml-starter", config={"n_estimators": 200, "max_depth": 8})
cfg = wandb.config

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(
    n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, random_state=42
)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

f1 = f1_score(y_test, pred)
wandb.log({"f1": f1})
wandb.finish()
