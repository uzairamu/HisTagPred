"""
HisTagPred — full training pipeline.
Reproduces the XGBoost model reported in the manuscript.

Usage:
    python train.py --embeddings X_esm_float32.npy
                    --labels     y_labels.npy
                    --output_dir models/
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (matthews_corrcoef, roc_auc_score,
                              accuracy_score, f1_score, make_scorer)
from xgboost import XGBClassifier


def run_cv(X, y, n_splits=5, random_state=42):
    """Run stratified 5-fold CV and report all metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)
    scoring = {
        "accuracy" : "accuracy",
        "f1"       : make_scorer(f1_score, pos_label=1),
        "mcc"      : make_scorer(matthews_corrcoef),
        "roc_auc"  : "roc_auc",
    }
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators    = 300,
            max_depth       = 4,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            eval_metric     = "logloss",
            scale_pos_weight= (y == 0).sum() / (y == 1).sum(),
            random_state    = random_state,
            n_jobs          = -1,
            tree_method     = "hist",
        )),
    ])
    results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring)
    print("\n=== 5-fold CV Results ===")
    for metric in ["accuracy", "f1", "mcc", "roc_auc"]:
        vals = results[f"test_{metric}"]
        print(f"  {metric:<12}: {vals.mean():.3f} +/- {vals.std():.3f}")
    return results


def train_final(X, y, output_dir, random_state=42):
    """Train final model on full dataset and save."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = XGBClassifier(
        n_estimators    = 300,
        max_depth       = 4,
        learning_rate   = 0.05,
        subsample       = 0.8,
        colsample_bytree= 0.8,
        eval_metric     = "logloss",
        scale_pos_weight= (y == 0).sum() / (y == 1).sum(),
        random_state    = random_state,
        n_jobs          = -1,
        tree_method     = "hist",
    )
    clf.fit(X_scaled, y)

    train_acc = (clf.predict(X_scaled) == y).mean()
    print(f"\nFinal model training accuracy: {train_acc:.3f}")

    model_path  = f"{output_dir}/xgboost_final.pkl"
    scaler_path = f"{output_dir}/scaler_final.pkl"
    joblib.dump(clf,    model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model  saved → {model_path}")
    print(f"Scaler saved → {scaler_path}")
    return clf, scaler


def main():
    parser = argparse.ArgumentParser(
        description="Train HisTagPred XGBoost classifier")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to ESM-2 embeddings .npy (N x 2560)")
    parser.add_argument("--labels",     type=str, required=True,
                        help="Path to labels .npy (N,) binary")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save trained model")
    parser.add_argument("--cv_only",    action="store_true",
                        help="Run CV only, do not save final model")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    X = np.load(args.embeddings)
    y = np.load(args.labels)
    print(f"X shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"C_terminal (0): {(y==0).sum()}  N_terminal (1): {(y==1).sum()}")

    run_cv(X, y, random_state=args.seed)

    if not args.cv_only:
        train_final(X, y, args.output_dir, random_state=args.seed)


if __name__ == "__main__":
    main()
