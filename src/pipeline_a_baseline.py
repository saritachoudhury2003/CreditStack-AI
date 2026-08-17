"""
pipeline_a_baseline.py
----------------------
Baseline XGBoost classifier with GridSearchCV tuning.
Exposes train_pipeline_a(X_train, X_test, y_train, y_test) for import.
Runnable standalone via __main__.
"""

import os
import time
import joblib
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

warnings.filterwarnings("ignore")


def train_pipeline_a(X_train, X_test, y_train, y_test):
    """
    Train a GridSearchCV-tuned XGBoost classifier and evaluate on test set.

    Parameters
    ----------
    X_train, X_test : DataFrames
    y_train, y_test : Series

    Returns
    -------
    model   : best XGBClassifier from GridSearchCV
    metrics : dict with keys accuracy, roc_auc, precision, recall, f1, training_time
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = round(neg / pos, 4)

    xgb_base = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=xgb_base,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    training_time = round(time.time() - t0, 2)

    best_model  = grid_search.best_estimator_
    y_pred      = best_model.predict(X_test)
    y_pred_prob = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":      round(accuracy_score(y_test, y_pred),  4),
        "roc_auc":       round(roc_auc_score(y_test, y_pred_prob), 4),
        "precision":     round(precision_score(y_test, y_pred), 4),
        "recall":        round(recall_score(y_test, y_pred),    4),
        "f1":            round(f1_score(y_test, y_pred),        4),
        "training_time": training_time,
    }

    return best_model, metrics


# ───────────────────────────────────────────────────────────────────
# Standalone execution
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed_data")
    MODELS_DIR    = os.path.join(BASE_DIR, "..", "models")
    DIVIDER       = "=" * 60

    print(DIVIDER)
    print("Loading preprocessed splits ...")
    print(DIVIDER)
    X_train = joblib.load(os.path.join(PROCESSED_DIR, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(PROCESSED_DIR, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    print(f"\n{DIVIDER}")
    print("Training Pipeline A (XGBoost + GridSearchCV) ...")
    print(DIVIDER)
    model, metrics = train_pipeline_a(X_train, X_test, y_train, y_test)

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*36}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:>10}")

    # Print detailed report
    y_pred = model.predict(X_test)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n  {cm}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "pipeline_a_model.pkl")
    joblib.dump(model, out_path)
    print(f"\n  Model saved to: {os.path.abspath(out_path)}")
    print(f"\n[DONE] Pipeline A complete!")
