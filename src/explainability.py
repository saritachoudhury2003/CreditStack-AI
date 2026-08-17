"""
explainability.py
-----------------
SHAP-based model explainability using KernelExplainer.
Exposes generate_shap_plots(model, X_test) for import into Streamlit.
Returns matplotlib Figure objects so they can be displayed anywhere.
Runnable standalone via __main__ (saves plots to results/).
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap
import joblib

warnings.filterwarnings("ignore")


def generate_shap_plots(model, X_test, n_explain: int = 40, nsamples: int = 80):
    """
    Generate SHAP summary and waterfall plots for a given model.

    Parameters
    ----------
    model     : trained sklearn-compatible classifier with predict_proba
    X_test    : pd.DataFrame — feature matrix (with column names)
    n_explain : number of test rows to explain (default 40 for speed)
    nsamples  : KernelExplainer approximation samples (default 80)

    Returns
    -------
    summary_fig   : matplotlib Figure — beeswarm summary plot
    waterfall_fig : matplotlib Figure — single-prediction waterfall
    shap_values   : np.ndarray — raw SHAP values for all explained rows
    feature_names : list[str]
    """
    feature_names = X_test.columns.tolist()
    X_explain = X_test.iloc[: min(n_explain, len(X_test))]

    # ── Prediction wrapper for KernelExplainer ──────────────────────
    def predict_fn(X_arr):
        df = pd.DataFrame(X_arr, columns=feature_names)
        return model.predict_proba(df)[:, 1]

    # ── Build explainer with k-means background ─────────────────────
    background = shap.kmeans(X_test.values, 10)
    explainer  = shap.KernelExplainer(predict_fn, background)

    # ── Compute SHAP values ─────────────────────────────────────────
    shap_values = explainer.shap_values(
        X_explain.values, nsamples=nsamples, silent=True
    )

    # ── Summary plot (beeswarm) ─────────────────────────────────────
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=len(feature_names),
        plot_size=None,
    )
    summary_fig = plt.gcf()
    summary_fig.patch.set_facecolor("#0f1117")
    ax = summary_fig.gca()
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="#d0d6e8")
    ax.set_xlabel(
        "SHAP value  (impact on P(Loan Approved))",
        fontsize=11, color="#9099b5",
    )
    summary_fig.suptitle(
        "SHAP Summary — Global Feature Importance",
        fontsize=13, color="#ffffff", fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    # ── Waterfall plot for one true-positive example ────────────────
    y_pred  = (model.predict_proba(X_explain)[:, 1] >= 0.5).astype(int)
    tp_mask = (y_pred == 1)
    idx     = int(np.argmax(tp_mask)) if tp_mask.any() else 0

    explanation = shap.Explanation(
        values        = shap_values[idx],
        base_values   = float(explainer.expected_value),
        data          = X_explain.iloc[idx].values,
        feature_names = feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=len(feature_names), show=False)
    waterfall_fig = plt.gcf()
    waterfall_fig.patch.set_facecolor("#0f1117")
    ax2 = waterfall_fig.gca()
    ax2.set_facecolor("#0f1117")
    ax2.tick_params(colors="#d0d6e8", labelsize=9)
    prob = model.predict_proba(X_explain.iloc[[idx]])[:, 1][0]
    waterfall_fig.suptitle(
        f"SHAP Waterfall — Test Sample #{idx}  (P(Approved)={prob:.3f})",
        fontsize=12, color="#ffffff", fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    return summary_fig, waterfall_fig, shap_values, feature_names


# ───────────────────────────────────────────────────────────────────
# Standalone execution — saves both plots to results/
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    matplotlib.use("Agg")

    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR    = os.path.join(BASE_DIR, "..", "models")
    PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed_data")
    RESULTS_DIR   = os.path.join(BASE_DIR, "..", "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading model and test data ...")
    model_b = joblib.load(os.path.join(MODELS_DIR,    "pipeline_b_model.pkl"))
    X_test  = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))

    print("Generating SHAP plots (n_explain=60, nsamples=100) ...")
    summary_fig, waterfall_fig, shap_vals, feat_names = generate_shap_plots(
        model_b, X_test, n_explain=60, nsamples=100
    )

    summary_path   = os.path.join(RESULTS_DIR, "shap_summary_plot.png")
    waterfall_path = os.path.join(RESULTS_DIR, "shap_waterfall_plot.png")
    summary_fig.savefig(summary_path,   dpi=150, bbox_inches="tight",
                        facecolor=summary_fig.get_facecolor())
    waterfall_fig.savefig(waterfall_path, dpi=150, bbox_inches="tight",
                          facecolor=waterfall_fig.get_facecolor())
    plt.close("all")

    print(f"  Summary plot  saved : {os.path.abspath(summary_path)}")
    print(f"  Waterfall plot saved: {os.path.abspath(waterfall_path)}")

    # Feature importance ranking
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = (
        pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.index += 1
    print("\nFeature Importance Ranking:")
    print(importance_df.to_string())
    print("\n[DONE] Explainability complete!")
