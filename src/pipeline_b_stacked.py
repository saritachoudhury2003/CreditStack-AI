"""
pipeline_b_stacked.py
---------------------
Stacking ensemble: RandomForest + XGBoost + LightGBM + SVC base learners,
LogisticRegression meta-learner.
Exposes train_pipeline_b(X_train, X_test, y_train, y_test) for import.
Runnable standalone via __main__.
"""

import os
import time
import joblib
import warnings
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


def train_pipeline_b(X_train, X_test, y_train, y_test):
    """
    Train a StackingClassifier (RF + XGBoost + LightGBM + SVC)
    with LogisticRegression as the meta-learner, and evaluate on test set.

    Parameters
    ----------
    X_train, X_test : DataFrames
    y_train, y_test : Series

    Returns
    -------
    model   : trained StackingClassifier
    metrics : dict with keys accuracy, roc_auc, precision, recall, f1, training_time
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = round(neg / pos, 4)

    base_estimators = [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight="balanced", random_state=42, n_jobs=-1,
            ),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                scale_pos_weight=scale_pos, objective="binary:logistic",
                eval_metric="logloss", random_state=42, n_jobs=-1,
            ),
        ),
        (
            "lgbm",
            LGBMClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            ),
        ),
        (
            "svc",
            SVC(
                kernel="rbf", C=1.0, gamma="scale",
                probability=True, class_weight="balanced", random_state=42,
            ),
        ),
    ]

    meta_learner = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=42,
    )

    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,
        stack_method="predict_proba",
        passthrough=False,
        n_jobs=-1,
    )

    t0 = time.time()
    stacking_model.fit(X_train, y_train)
    training_time = round(time.time() - t0, 2)

    y_pred      = stacking_model.predict(X_test)
    y_pred_prob = stacking_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":      round(accuracy_score(y_test, y_pred),  4),
        "roc_auc":       round(roc_auc_score(y_test, y_pred_prob), 4),
        "precision":     round(precision_score(y_test, y_pred), 4),
        "recall":        round(recall_score(y_test, y_pred),    4),
        "f1":            round(f1_score(y_test, y_pred),        4),
        "training_time": training_time,
    }

    return stacking_model, metrics


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
    print("Training Pipeline B (Stacking Ensemble) ...")
    print(DIVIDER)
    model, metrics = train_pipeline_b(X_train, X_test, y_train, y_test)

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*36}")
    for k, v in metrics.items():
        print(f"  {k:<25} {v:>10}")

    y_pred = model.predict(X_test)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Rejected', 'Approved'])}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n  {cm}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, "pipeline_b_model.pkl")
    joblib.dump(model, out_path)
    print(f"\n  Model saved to: {os.path.abspath(out_path)}")
    print(f"\n[DONE] Pipeline B complete!")
