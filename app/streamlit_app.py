"""
streamlit_app.py
----------------
CreditStack AI — Multi-model credit risk scoring comparison tool.
Run with: streamlit run app/streamlit_app.py
"""

import sys
import os

# ── Make src/ and agent/ importable regardless of working directory ───────────
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "src")))
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from preprocessing      import preprocess_data
from pipeline_a_baseline import train_pipeline_a
from pipeline_b_stacked  import train_pipeline_b
from explainability      import generate_shap_plots

from agent.orchestrator import run_agent_pipeline
from agent.underwriter_agent import run_underwriter_agent

# ───────────────────────────────────────────────────────────────────
# Page config
# ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditStack AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ───────────────────────────────────────────────────────────────────
# Custom CSS — dark premium theme
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Global ── */
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0a0c14 0%, #0f1520 60%, #0a1128 100%);
    color: #d0d6e8;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #1a2040 0%, #0e1830 50%, #111828 100%);
    border: 1px solid #2a3560;
    border-radius: 16px;
    padding: 2.4rem 2.8rem 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 40px rgba(79,142,247,0.12);
  }
  .hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4f8ef7, #a78bfa, #f76b4f);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
  }
  .hero p {
    font-size: 1.05rem;
    color: #8892b0;
    margin: 0;
    line-height: 1.6;
  }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #a8c4ff;
    border-left: 4px solid #4f8ef7;
    padding-left: 0.75rem;
    margin: 2rem 0 1rem;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(135deg, #161c2e, #1a2240);
    border: 1px solid #2a3560;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(79,142,247,0.2);
  }
  .metric-label { font-size: 0.78rem; color: #6b7a99; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: #4f8ef7; margin: 0.2rem 0; }
  .metric-value.orange { color: #f76b4f; }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: #141b2d;
    border: 2px dashed #2a3560;
    border-radius: 12px;
    padding: 1rem;
  }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* ── Spinner ── */
  [data-testid="stSpinner"] > div { color: #4f8ef7 !important; }

  /* ── Alert / info ── */
  [data-testid="stAlert"] { border-radius: 10px; }

  /* ── Divider ── */
  hr { border-color: #1e2740; margin: 2rem 0; }

  /* ── Badge ── */
  .badge {
    display: inline-block;
    background: #1a2a50;
    border: 1px solid #2a4080;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: #7099dd;
    margin-right: 0.4rem;
  }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────
# Hero banner
# ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>💳 CreditStack AI</h1>
  <p>
    A multi-model credit risk scoring platform that trains, compares, and
    explains two ML pipelines on your loan dataset — a tuned
    <strong>XGBoost baseline</strong> (Pipeline A) versus a
    <strong>stacking ensemble</strong> of RF + XGBoost + LightGBM + SVC
    (Pipeline B) — with full SHAP explainability for every prediction.
  </p>
  <br/>
  <span class="badge">XGBoost</span>
  <span class="badge">LightGBM</span>
  <span class="badge">Stacking Ensemble</span>
  <span class="badge">SHAP Explainability</span>
  <span class="badge">GridSearchCV</span>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────────
# Section 1 — File Upload
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">1 &nbsp; Upload Your Dataset</div>', unsafe_allow_html=True)

st.markdown(
    "Upload a CSV matching the **loan_data.csv** schema. "
    "Required columns: `Loan_ID`, `Gender`, `Married`, `Dependents`, `Education`, "
    "`Self_Employed`, `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, "
    "`Loan_Amount_Term`, `Credit_History`, `Property_Area`, `Loan_Status`."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="CSV must match the loan_data.csv column structure.",
)

if uploaded_file is None:
    st.info("Upload a CSV file above to start the analysis.")
    st.stop()

# ───────────────────────────────────────────────────────────────────
# Load and validate uploaded CSV
# ───────────────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file as a CSV: {e}")
    st.stop()

# ───────────────────────────────────────────────────────────────────
# Section 2 — Data Preview
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">2 &nbsp; Data Preview</div>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Total Rows</div>
      <div class="metric-value">{raw_df.shape[0]:,}</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Total Columns</div>
      <div class="metric-value">{raw_df.shape[1]}</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    missing = raw_df.isnull().sum().sum()
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Missing Cells</div>
      <div class="metric-value {'orange' if missing > 0 else ''}">{missing}</div>
    </div>""", unsafe_allow_html=True)

st.dataframe(raw_df.head(10), use_container_width=True)

# ── Missing value summary ───────────────────────────────────────────
if raw_df.isnull().any().any():
    with st.expander("Missing value summary"):
        miss_df = (
            raw_df.isnull().sum()
            .reset_index()
            .rename(columns={"index": "Column", 0: "Missing Count"})
        )
        miss_df = miss_df[miss_df["Missing Count"] > 0]
        st.dataframe(miss_df, use_container_width=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 3 — Preprocessing
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">3 &nbsp; Preprocessing & Feature Engineering</div>', unsafe_allow_html=True)

try:
    with st.spinner("Running preprocessing pipeline..."):
        X_train, X_test, y_train, y_test = preprocess_data(raw_df)
except ValueError as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error during preprocessing: {e}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.success(f"Training set: **{X_train.shape[0]} rows x {X_train.shape[1]} features**")
with col2:
    st.success(f"Test set: **{X_test.shape[0]} rows x {X_test.shape[1]} features**")

with st.expander("Engineered feature list"):
    st.write(X_train.columns.tolist())

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 4 — Train both pipelines
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">4 &nbsp; Model Training</div>', unsafe_allow_html=True)

model_a, model_b, metrics_a, metrics_b = None, None, None, None

with st.spinner("Training Pipeline A — XGBoost + GridSearchCV (this may take ~30s)..."):
    try:
        model_a, metrics_a = train_pipeline_a(X_train, X_test, y_train, y_test)
    except Exception as e:
        st.error(f"Pipeline A training failed: {e}")
        st.stop()
st.success("Pipeline A trained!")

with st.spinner("Training Pipeline B — Stacking Ensemble (RF + XGBoost + LightGBM + SVC, ~30s)..."):
    try:
        model_b, metrics_b = train_pipeline_b(X_train, X_test, y_train, y_test)
    except Exception as e:
        st.error(f"Pipeline B training failed: {e}")
        st.stop()
st.success("Pipeline B trained!")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 5 — Pipeline Comparison
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">5 &nbsp; Pipeline Comparison</div>', unsafe_allow_html=True)

METRIC_LABELS = {
    "accuracy":      "Accuracy",
    "roc_auc":       "ROC-AUC",
    "precision":     "Precision",
    "recall":        "Recall",
    "f1":            "F1-Score",
    "training_time": "Train Time (s)",
}

# ── Comparison table ────────────────────────────────────────────────
comparison_df = pd.DataFrame(
    {
        "Metric":               [METRIC_LABELS[k] for k in METRIC_LABELS],
        "Pipeline A (XGBoost)": [metrics_a[k] for k in METRIC_LABELS],
        "Pipeline B (Stacking)":[metrics_b[k] for k in METRIC_LABELS],
        "Delta (B - A)":        [round(metrics_b[k] - metrics_a[k], 4) for k in METRIC_LABELS],
    }
)
st.dataframe(comparison_df.set_index("Metric"), use_container_width=True)

# ── Metric cards row ────────────────────────────────────────────────
st.markdown("**Pipeline A — XGBoost Baseline**")
cols = st.columns(5)
metric_keys = ["accuracy", "roc_auc", "precision", "recall", "f1"]
for col, key in zip(cols, metric_keys):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{METRIC_LABELS[key]}</div>
          <div class="metric-value">{metrics_a[key]:.4f}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("**Pipeline B — Stacking Ensemble**")
cols2 = st.columns(5)
for col, key in zip(cols2, metric_keys):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{METRIC_LABELS[key]}</div>
          <div class="metric-value orange">{metrics_b[key]:.4f}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ── Bar chart ───────────────────────────────────────────────────────
st.subheader("Metric Comparison Chart")

PLOT_KEYS   = ["accuracy", "roc_auc", "precision", "recall", "f1"]
PLOT_LABELS = [METRIC_LABELS[k] for k in PLOT_KEYS]
vals_a = [metrics_a[k] for k in PLOT_KEYS]
vals_b = [metrics_b[k] for k in PLOT_KEYS]

x     = np.arange(len(PLOT_LABELS))
width = 0.32

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars_a = ax.bar(x - width / 2, vals_a, width, label="Pipeline A — XGBoost",
                color="#4f8ef7", edgecolor="#2a5abf", linewidth=0.8, zorder=3)
bars_b = ax.bar(x + width / 2, vals_b, width, label="Pipeline B — Stacking",
                color="#f76b4f", edgecolor="#bf3a2a", linewidth=0.8, zorder=3)

for bar in bars_a:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{bar.get_height():.4f}", ha="center", va="bottom",
            fontsize=8.5, color="#a8c4ff", fontweight="bold")
for bar in bars_b:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
            f"{bar.get_height():.4f}", ha="center", va="bottom",
            fontsize=8.5, color="#ffc4b8", fontweight="bold")

ax.set_ylim(0.55, 1.00)
ax.set_xticks(x)
ax.set_xticklabels(PLOT_LABELS, fontsize=11, color="#d0d6e8")
ax.tick_params(axis="y", colors="#d0d6e8", labelsize=10)
ax.set_ylabel("Score", fontsize=12, color="#9099b5")
ax.set_title("Pipeline A vs Pipeline B — Test Set Metrics",
             fontsize=13, color="#ffffff", fontweight="bold", pad=14)
ax.legend(fontsize=10, facecolor="#262a3a", edgecolor="#444860",
          labelcolor="#d0d6e8", loc="lower right")
ax.grid(axis="y", color="#2e3347", linewidth=0.7, zorder=0)
for spine in ax.spines.values():
    spine.set_edgecolor("#2e3347")

# Training time annotation
ax.text(0.01, 0.97,
        f"Train Times  |  A: {metrics_a['training_time']}s   B: {metrics_b['training_time']}s",
        transform=ax.transAxes, fontsize=9, va="top", color="#9099b5",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e2233",
                  edgecolor="#3a3f58", alpha=0.9))

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 6 — Model Explainability (SHAP)
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">6 &nbsp; Model Explainability — Pipeline B (SHAP)</div>',
            unsafe_allow_html=True)

st.markdown(
    "Using **SHAP KernelExplainer** (model-agnostic) to explain the stacking "
    "ensemble's predictions. This reveals which features drove each decision "
    "and by how much."
)

with st.spinner("Computing SHAP values (explaining 40 test samples — ~60s)..."):
    try:
        summary_fig, waterfall_fig, shap_vals, feat_names = generate_shap_plots(
            model_b, X_test, n_explain=40, nsamples=80
        )
    except Exception as e:
        st.error(f"SHAP explainability failed: {e}")
        st.stop()

st.subheader("SHAP Summary Plot — Global Feature Importance")
st.markdown(
    "Each dot represents one test example. The **x-axis** shows SHAP value "
    "(positive = pushes toward 'Approved'). **Colour** = feature value "
    "(red = high, blue = low)."
)
st.pyplot(summary_fig)
plt.close(summary_fig)

st.subheader("SHAP Waterfall Plot — Single Prediction Breakdown")
st.markdown(
    "Shows how each feature pushed the prediction above or below the "
    "base approval rate for one specific applicant."
)
st.pyplot(waterfall_fig)
plt.close(waterfall_fig)

# ── Feature importance table ─────────────────────────────────────
mean_abs = np.abs(shap_vals).mean(axis=0)
importance_df = (
    pd.DataFrame({"Feature": feat_names, "Mean |SHAP Value|": mean_abs})
    .sort_values("Mean |SHAP Value|", ascending=False)
    .reset_index(drop=True)
)
importance_df.index += 1

with st.expander("Full feature importance ranking"):
    st.dataframe(importance_df, use_container_width=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 7 — Verdict
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">7 &nbsp; Verdict & Dynamic Recommendation</div>',
            unsafe_allow_html=True)

# Define logic for dynamic recommendation
# In credit risk, precision (avoiding false approvals) and ROC-AUC are usually key.
score_a = metrics_a["roc_auc"] * 0.6 + metrics_a["precision"] * 0.4
score_b = metrics_b["roc_auc"] * 0.6 + metrics_b["precision"] * 0.4

if score_b > score_a:
    recommended_model = "Pipeline B (Stacking Ensemble)"
    other_model = "Pipeline A (XGBoost Baseline)"
    reasons = [
        f"**Higher discriminative power**: It achieved a better ROC-AUC ({metrics_b['roc_auc']:.4f} vs {metrics_a['roc_auc']:.4f}).",
        f"**More conservative approvals**: It achieved higher precision ({metrics_b['precision']:.4f} vs {metrics_a['precision']:.4f}), which is critical for reducing default risk.",
    ]
    if metrics_b["f1"] > metrics_a["f1"]:
        reasons.append(f"**Better overall balance**: It achieved a higher F1-score ({metrics_b['f1']:.4f} vs {metrics_a['f1']:.4f}).")
else:
    recommended_model = "Pipeline A (XGBoost Baseline)"
    other_model = "Pipeline B (Stacking Ensemble)"
    reasons = [
        f"**Higher discriminative power**: It achieved a better ROC-AUC ({metrics_a['roc_auc']:.4f} vs {metrics_b['roc_auc']:.4f}).",
        f"**More conservative approvals**: It achieved higher precision ({metrics_a['precision']:.4f} vs {metrics_b['precision']:.4f}), reducing default risk.",
    ]
    if metrics_a["f1"] > metrics_b["f1"]:
        reasons.append(f"**Better overall balance**: It achieved a higher F1-score ({metrics_a['f1']:.4f} vs {metrics_b['f1']:.4f}).")

st.success(f"### Recommended Model: {recommended_model}")
st.write("Based on the evaluation metrics, this model is recommended for production credit risk scoring because:")
for reason in reasons:
    st.markdown(f"- {reason}")

with st.expander("Why might I use the other model?"):
    st.write(f"You might still consider **{other_model}** if:")
    if recommended_model == "Pipeline B (Stacking Ensemble)":
        st.markdown(f"- You need faster training times ({metrics_a['training_time']}s vs {metrics_b['training_time']}s).")
        st.markdown("- You prefer a simpler, more interpretable single model over a complex ensemble.")
        if metrics_a["recall"] > metrics_b["recall"]:
             st.markdown(f"- You want to maximize the number of approved loans (higher recall: {metrics_a['recall']:.4f} vs {metrics_b['recall']:.4f}).")
    else:
        st.markdown("- You want to leverage a diverse ensemble of models.")
        if metrics_b["recall"] > metrics_a["recall"]:
             st.markdown(f"- You want to maximize the number of approved loans (higher recall: {metrics_b['recall']:.4f} vs {metrics_a['recall']:.4f}).")

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 8 — AI Multi-Agent Orchestrator
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">8 &nbsp; AI Multi-Agent Orchestrator</div>', unsafe_allow_html=True)
st.markdown(
    "Select an applicant from the test set below to simulate our **Agentic Workflow**. "
    "The Risk Analyst, Compliance Agent, and Communication Agent coordinate to analyze risk and draft a fair letter."
)

# Applicant Selector Dropdown
max_applicants = min(len(X_test), len(shap_vals))
applicant_options = [f"Applicant #{i} (Index {i})" for i in range(max_applicants)]

selected_applicant_str = st.selectbox(
    "Choose Candidate to Analyze:", 
    options=applicant_options, 
    index=0,
    key="selected_applicant_select"
)
selected_idx = int(selected_applicant_str.split("#")[1].split(" ")[0])

# Reset chat history if user switches candidate
if "last_selected_idx" not in st.session_state:
    st.session_state.last_selected_idx = selected_idx
elif st.session_state.last_selected_idx != selected_idx:
    st.session_state.last_selected_idx = selected_idx
    st.session_state.orchestrator_run = False
    st.session_state.chat_history = []

if "orchestrator_run" not in st.session_state:
    st.session_state.orchestrator_run = False

if st.button(f"Run Multi-Agent Pipeline for Applicant #{selected_idx}"):
    with st.spinner("Agents are collaborating (this involves multiple LLM calls, please wait ~15s)..."):
        try:
            applicant_shap = shap_vals[selected_idx]
            
            # Reconstruct the row for predict_proba
            df_app = pd.DataFrame([X_test.iloc[selected_idx].values], columns=feat_names)
            prob_val = float(model_b.predict_proba(df_app)[:, 1][0])
            
            decision = "APPROVE" if prob_val >= 0.5 else "REJECT"
            business_rule = "Standard Algorithmic Review"
            
            pipeline_result = run_agent_pipeline(
                probability=prob_val,
                shap_values=list(applicant_shap),
                feature_names=feat_names,
                business_rule=business_rule,
                decision=decision
            )
            st.session_state.pipeline_result = pipeline_result
            st.session_state.orchestrator_run = True
            st.session_state.applicant_prob = prob_val
            st.session_state.applicant_shap = list(applicant_shap)
            
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

if st.session_state.orchestrator_run:
    result = st.session_state.pipeline_result
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🕵️ Risk Analyst Summary")
        st.info(result.get("risk_summary", "No summary generated."))
        
        st.subheader("⚖️ Compliance Check")
        status = result.get("compliance_status", "ERROR")
        if status == "APPROVED":
            st.success(f"**Status:** {status}\n\n**Notes:** {result.get('compliance_notes', '')}")
        else:
            st.error(f"**Status:** {status}\n\n**Notes:** {result.get('compliance_notes', '')}")
            
    with col2:
        st.subheader("✉️ Final Applicant Letter")
        st.text_area("Generated by Communication Agent", value=result.get("final_letter", ""), height=250, disabled=True)

st.markdown("---")

# ───────────────────────────────────────────────────────────────────
# Section 9 — Interactive Underwriter Assistant
# ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">9 &nbsp; Interactive Underwriter Assistant</div>', unsafe_allow_html=True)
st.markdown(
    f"Chat with the **Underwriter Agent** about **Applicant #{selected_idx}**. It has tools to fetch document status, "
    "explain SHAP values, and simulate alternate income/DTI scenarios."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Inline Form for Querying the Underwriter Agent
with st.form(key="underwriter_chat_form", clear_on_submit=True):
    user_query = st.text_input(f"Ask the Underwriter Agent about Applicant #{selected_idx}:", placeholder="e.g., What are the main risk factors? Or simulate new income $8000 and DTI 0.15")
    submit_button = st.form_submit_button("Ask Agent 🤖")

if submit_button and user_query:
    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Build context for the selected applicant
    app_context = {
        "applicant_id": f"APP-{selected_idx:03d}",
        "probability": float(model_b.predict_proba(pd.DataFrame([X_test.iloc[selected_idx].values], columns=feat_names))[:, 1][0]),
        "shap_values": list(shap_vals[selected_idx]),
        "feature_names": feat_names,
        "base_data": X_test.iloc[selected_idx].to_dict()
    }

    # Call agent
    with st.spinner("Agent is thinking (may use tools to fetch docs or simulate scenarios)..."):
        response = run_underwriter_agent(user_query, app_context)
        
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history below the form
if st.session_state.chat_history:
    st.markdown("### Conversation History")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

st.markdown(
    "<br/><center style='color:#444;font-size:0.8rem;'>CreditStack AI — Built with Streamlit, XGBoost, LightGBM, scikit-learn & Ollama</center>",
    unsafe_allow_html=True,
)
