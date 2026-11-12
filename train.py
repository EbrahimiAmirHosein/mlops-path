import argparse
import json
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_data(seed: int = 42):
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)


def build_model(C: float = 1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    X_tr, X_te, y_tr, y_te = load_data()
    model = build_model(args.C).fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out / "model.joblib")
    (args.out / "metrics.json").write_text(json.dumps({"AUC is": auc}))
    print(f"AUC={auc:.2f}")


if __name__ == "__main__":
    main()
