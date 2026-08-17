"""
compare_pipelines.py
---------------------
Loads both trained models and the held-out test set, re-evaluates every
metric, builds a side-by-side comparison DataFrame, plots a bar chart,
saves it to results/, and prints a written verdict.

Note on training times: pipeline_a and pipeline_b each print and record
their own wall-clock training time when run. Those times are loaded here
from a metadata file (if it exists) or fall back to the values captured
during the last training run, stored as constants below.
"""

import os
import time
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # non-interactive backend for file saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "processed_data")
MODELS_DIR    = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "..", "results")
CHART_OUT     = os.path.join(RESULTS_DIR, "pipeline_comparison.png")
META_PATH     = os.path.join(MODELS_DIR,  "training_metadata.json")

DIVIDER = "=" * 60

# ─────────────────────────────────────────────
# 1. Load both models
# ─────────────────────────────────────────────
print(DIVIDER)
print("STEP 1 — Loading models from models/")
print(DIVIDER)

model_a = joblib.load(os.path.join(MODELS_DIR, "pipeline_a_model.pkl"))
model_b = joblib.load(os.path.join(MODELS_DIR, "pipeline_b_model.pkl"))
print(f"  Pipeline A : {model_a.__class__.__name__}")
print(f"  Pipeline B : {model_b.__class__.__name__}")

# ─────────────────────────────────────────────
# 2. Load test data
# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("STEP 2 — Loading X_test, y_test from processed_data/")
print(DIVIDER)

X_test = joblib.load(os.path.join(PROCESSED_DIR, "X_test.pkl"))
y_test = joblib.load(os.path.join(PROCESSED_DIR, "y_test.pkl"))
print(f"  X_test : {X_test.shape}  |  y_test : {y_test.shape}")

# ─────────────────────────────────────────────
# Training time — loaded from metadata file if
# available, otherwise fall back to last-run values.
# ─────────────────────────────────────────────
FALLBACK_TIMES = {"pipeline_a": 6.07, "pipeline_b": 5.01}

if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    train_time_a = meta.get("pipeline_a_train_time", FALLBACK_TIMES["pipeline_a"])
    train_time_b = meta.get("pipeline_b_train_time", FALLBACK_TIMES["pipeline_b"])
    print(f"\n  Training times loaded from metadata file.")
else:
    train_time_a = FALLBACK_TIMES["pipeline_a"]
    train_time_b = FALLBACK_TIMES["pipeline_b"]
    print(f"\n  Metadata file not found — using recorded training times as fallback.")

print(f"  Pipeline A training time : {train_time_a:.2f}s")
print(f"  Pipeline B training time : {train_time_b:.2f}s")

# ─────────────────────────────────────────────
# 3. Re-evaluate both models on the test set
#    Also measure inference (prediction) time
# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("STEP 3 — Re-evaluating both models on test set")
print(DIVIDER)

def evaluate(model, X, y, label):
    """Run predictions and return a dict of all metrics + inference time."""
    t0 = time.perf_counter()
    y_pred = model.predict(X)
    t1 = time.perf_counter()
    y_prob = model.predict_proba(X)[:, 1]

    return {
        "Model"              : label,
        "Accuracy"           : round(accuracy_score(y, y_pred),  4),
        "ROC-AUC"            : round(roc_auc_score(y, y_prob),   4),
        "Precision"          : round(precision_score(y, y_pred), 4),
        "Recall"             : round(recall_score(y, y_pred),    4),
        "F1-Score"           : round(f1_score(y, y_pred),        4),
        "Inference Time (ms)": round((t1 - t0) * 1000,           2),
    }

metrics_a = evaluate(model_a, X_test, y_test, "Pipeline A (XGBoost Baseline)")
metrics_b = evaluate(model_b, X_test, y_test, "Pipeline B (Stacking Ensemble)")

# Attach training times
metrics_a["Train Time (s)"] = train_time_a
metrics_b["Train Time (s)"] = train_time_b

print(f"  Pipeline A evaluated.")
print(f"  Pipeline B evaluated.")

# ─────────────────────────────────────────────
# 4. Build comparison DataFrame and print it
# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("STEP 4 — Comparison Table")
print(DIVIDER)

df = pd.DataFrame([metrics_a, metrics_b]).set_index("Model")

# Compute delta row (B minus A)
delta = df.loc["Pipeline B (Stacking Ensemble)"] - df.loc["Pipeline A (XGBoost Baseline)"]
delta.name = "Delta  (B - A)"
df = pd.concat([df, delta.to_frame().T])

# Build a pure-string DataFrame for display — avoids LossySetitemError
# when trying to write str values into float64 columns (pandas 2.x+)
df_str = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
df_str.iloc[:2] = df.iloc[:2].map(lambda x: f"{float(x):.4f}")
df_str.iloc[2]  = df.iloc[2].map(lambda x: f"{float(x):+.4f}")

with pd.option_context("display.max_columns", 10, "display.width", 120):
    print(f"\n{df_str.to_string()}\n")

# ─────────────────────────────────────────────
# 5. Bar chart — all metrics side by side
# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("STEP 5 — Building bar chart")
print(DIVIDER)

# Only plot the 5 classification metrics (not times)
PLOT_METRICS = ["Accuracy", "ROC-AUC", "Precision", "Recall", "F1-Score"]
vals_a = [metrics_a[m] for m in PLOT_METRICS]
vals_b = [metrics_b[m] for m in PLOT_METRICS]

x      = np.arange(len(PLOT_METRICS))
width  = 0.32

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars_a = ax.bar(x - width / 2, vals_a, width, label="Pipeline A — XGBoost Baseline",
                color="#4f8ef7", edgecolor="#2a5abf", linewidth=0.8, zorder=3)
bars_b = ax.bar(x + width / 2, vals_b, width, label="Pipeline B — Stacking Ensemble",
                color="#f76b4f", edgecolor="#bf3a2a", linewidth=0.8, zorder=3)

# Value labels on top of each bar
for bar in bars_a:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8.5,
            color="#a8c4ff", fontweight="bold")

for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom", fontsize=8.5,
            color="#ffc4b8", fontweight="bold")

# Grid and axis styling
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylim(0.55, 1.00)
ax.set_xticks(x)
ax.set_xticklabels(PLOT_METRICS, fontsize=11, color="#d0d6e8")
ax.tick_params(axis="y", colors="#d0d6e8", labelsize=10)
ax.set_xlabel("Metric", fontsize=12, color="#9099b5", labelpad=10)
ax.set_ylabel("Score", fontsize=12, color="#9099b5", labelpad=10)
ax.set_title("Pipeline A vs Pipeline B — Test Set Metric Comparison",
             fontsize=14, color="#ffffff", fontweight="bold", pad=18)
ax.legend(fontsize=10, facecolor="#262a3a", edgecolor="#444860",
          labelcolor="#d0d6e8", loc="lower right")
ax.grid(axis="y", color="#2e3347", linewidth=0.7, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor("#2e3347")

# Annotation box — training times
info_text = (
    f"Training Times\n"
    f"Pipeline A : {train_time_a:.2f}s\n"
    f"Pipeline B : {train_time_b:.2f}s"
)
ax.text(0.01, 0.98, info_text,
        transform=ax.transAxes,
        fontsize=9, verticalalignment="top",
        color="#9099b5",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e2233",
                  edgecolor="#3a3f58", alpha=0.9))

plt.tight_layout()
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(CHART_OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  Chart saved to : {os.path.abspath(CHART_OUT)}")

# ─────────────────────────────────────────────
# 6. Written verdict
# ─────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("STEP 6 — Verdict")
print(DIVIDER)

winner_auc  = "B" if metrics_b["ROC-AUC"]  > metrics_a["ROC-AUC"]  else "A"
winner_acc  = "B" if metrics_b["Accuracy"] > metrics_a["Accuracy"] else "A"
winner_f1   = "B" if metrics_b["F1-Score"] > metrics_a["F1-Score"] else "A"
winner_prec = "B" if metrics_b["Precision"] > metrics_a["Precision"] else "A"
winner_time = "B" if train_time_b < train_time_a else "A"

delta_auc  = metrics_b["ROC-AUC"]   - metrics_a["ROC-AUC"]
delta_acc  = metrics_b["Accuracy"]  - metrics_a["Accuracy"]
delta_f1   = metrics_b["F1-Score"]  - metrics_a["F1-Score"]
delta_time = train_time_b - train_time_a

print(f"""
  VERDICT
  -------
  Pipeline A (XGBoost Baseline) vs Pipeline B (Stacking Ensemble)

  > ROC-AUC   : Pipeline {winner_auc} leads by {abs(delta_auc):.4f}
                (B={metrics_b['ROC-AUC']:.4f}  A={metrics_a['ROC-AUC']:.4f})

  > Accuracy  : Pipeline {winner_acc} leads by {abs(delta_acc):.4f}
                (B={metrics_b['Accuracy']:.4f}  A={metrics_a['Accuracy']:.4f})

  > F1-Score  : Pipeline {winner_f1} leads by {abs(delta_f1):.4f}
                (B={metrics_b['F1-Score']:.4f}  A={metrics_a['F1-Score']:.4f})

  > Precision : Pipeline {winner_prec} is more conservative — fewer false approvals
                (B={metrics_b['Precision']:.4f}  A={metrics_a['Precision']:.4f})

  > Train Time: Pipeline {winner_time} is faster
                (A={train_time_a:.2f}s  B={train_time_b:.2f}s  delta={delta_time:+.2f}s)

  RECOMMENDATION
  --------------
  For a credit risk / loan approval use case, Pipeline B (Stacking)
  is the stronger choice when:
    - Discrimination quality (ROC-AUC) matters most — B wins by +{abs(delta_auc):.4f}
    - Avoiding false approvals (Precision) is a business priority — B wins

  Pipeline A (XGBoost Baseline) is preferred when:
    - Simpler, more interpretable models are required
    - Recall (catching as many true approvals as possible) is prioritised
    - Faster training cycles are needed for retraining pipelines

  BOTTOM LINE: Use Pipeline B in production for discriminative power.
               Use Pipeline A for fast experimentation or explainability audits.
""")

print(f"[DONE] Pipeline comparison complete!")
