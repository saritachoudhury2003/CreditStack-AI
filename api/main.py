"""
api/main.py
-----------
FastAPI backend server for CreditStack AI.
Provides REST APIs for model metrics, candidate data, and SSE streaming for agents.
"""

import sys
import os
import json
import asyncio
import pandas as pd
import numpy as np
import joblib
import shap

import io
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

# ── Ensure src/ and root are in sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, "src")))
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))

from preprocessing import preprocess_data
from pipeline_a_baseline import train_pipeline_a
from pipeline_b_stacked import train_pipeline_b
from explainability import generate_shap_plots

from agent.risk_analyst_agent import run_risk_analyst_agent
from agent.compliance_agent import run_compliance_agent
from agent.communication_agent import run_communication_agent
from agent.underwriter_agent import run_underwriter_agent
from agent import tools
from db.database import init_db, get_conn

app = FastAPI(title="CreditStack AI API", version="2.0")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize SQLite on startup ──────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()

# Global cached variables
DATA = {}

def get_processed_data():
    """Loads and preprocesses the dataset if not already loaded."""
    if "X_test" not in DATA:
        data_path = os.path.join(PROJECT_ROOT, "data", "loan_data.csv")
        df = pd.read_csv(data_path)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Load pre-trained models if available, else train
        model_a_path = os.path.join(PROJECT_ROOT, "models", "pipeline_a_model.pkl")
        model_b_path = os.path.join(PROJECT_ROOT, "models", "pipeline_b_model.pkl")
        
        if os.path.exists(model_a_path) and os.path.exists(model_b_path):
            model_a = joblib.load(model_a_path)
            model_b = joblib.load(model_b_path)
        else:
            model_a, _ = train_pipeline_a(X_train, X_test, y_train, y_test)
            model_b, _ = train_pipeline_b(X_train, X_test, y_train, y_test)
            
        # Compute SHAP values for X_test
        feature_names = list(X_test.columns)
        background = shap.kmeans(X_test.values, 10)
        
        def predict_fn(X_arr):
            tmp_df = pd.DataFrame(X_arr, columns=feature_names)
            return model_b.predict_proba(tmp_df)[:, 1]
            
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_test.values, nsamples=100, silent=True)[0]
        
        DATA["X_test"] = X_test
        DATA["y_test"] = y_test
        DATA["model_a"] = model_a
        DATA["model_b"] = model_b
        DATA["feature_names"] = feature_names
        DATA["shap_vals"] = shap_vals
        DATA["raw_df"] = df
        
    return DATA

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CreditStack AI API"}

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Accepts a user-uploaded CSV file, processes it, retrains/re-evaluates models,
    and updates global DATA context for Next.js dashboard.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Preprocess uploaded dataset
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Re-train models on new dataset
        model_a, _ = train_pipeline_a(X_train, X_test, y_train, y_test)
        model_b, _ = train_pipeline_b(X_train, X_test, y_train, y_test)
        
        # Compute SHAP values for X_test
        feature_names = list(X_test.columns)
        background = shap.kmeans(X_test.values, 10)
        
        def predict_fn(X_arr):
            tmp_df = pd.DataFrame(X_arr, columns=feature_names)
            return model_b.predict_proba(tmp_df)[:, 1]
            
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(X_test.values, nsamples=100, silent=True)[0]
        
        # Update global DATA
        DATA["X_test"] = X_test
        DATA["y_test"] = y_test
        DATA["model_a"] = model_a
        DATA["model_b"] = model_b
        DATA["feature_names"] = feature_names
        DATA["shap_vals"] = shap_vals
        DATA["raw_df"] = df
        
        # ── Save dataset upload record ─────────────────────────────────────────
        conn = get_conn()
        conn.execute(
            "INSERT INTO uploaded_datasets (filename, row_count) VALUES (?, ?)",
            (file.filename, len(df))
        )
        conn.commit()
        conn.close()
        # ─────────────────────────────────────────────────────────────────────

        return {
            "status": "success",
            "message": f"Successfully processed '{file.filename}' with {len(df)} rows.",
            "candidates_count": len(X_test)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process CSV: {str(e)}")

@app.get("/api/metrics")
def get_metrics():
    """Returns model comparison metrics for Pipeline A vs B."""
    d = get_processed_data()
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    
    # Evaluate Pipeline A
    y_pred_a = d["model_a"].predict(d["X_test"])
    y_prob_a = d["model_a"].predict_proba(d["X_test"])[:, 1]
    metrics_a = {
        "accuracy": float(accuracy_score(d["y_test"], y_pred_a)),
        "roc_auc": float(roc_auc_score(d["y_test"], y_prob_a)),
        "precision": float(precision_score(d["y_test"], y_pred_a)),
        "recall": float(recall_score(d["y_test"], y_pred_a)),
        "f1": float(f1_score(d["y_test"], y_pred_a)),
    }
    
    # Evaluate Pipeline B
    y_pred_b = d["model_b"].predict(d["X_test"])
    y_prob_b = d["model_b"].predict_proba(d["X_test"])[:, 1]
    metrics_b = {
        "accuracy": float(accuracy_score(d["y_test"], y_pred_b)),
        "roc_auc": float(roc_auc_score(d["y_test"], y_prob_b)),
        "precision": float(precision_score(d["y_test"], y_pred_b)),
        "recall": float(recall_score(d["y_test"], y_pred_b)),
        "f1": float(f1_score(d["y_test"], y_pred_b)),
    }
    
    return {
        "pipeline_a": metrics_a,
        "pipeline_b": metrics_b,
        "recommendation": "Pipeline B (Stacking Ensemble)" if metrics_b["roc_auc"] > metrics_a["roc_auc"] else "Pipeline A (XGBoost)"
    }

@app.get("/api/candidates")
def get_candidates():
    """Returns the list of candidates available for testing."""
    d = get_processed_data()
    count = len(d["X_test"])
    candidates = []
    for i in range(count):
        candidates.append({
            "index": i,
            "id": f"Applicant #{i} (APP-{i:03d})",
            "label": f"Applicant #{i}"
        })
    return {"candidates": candidates}

@app.get("/api/applicant/{idx}")
def get_applicant_detail(idx: int):
    """Returns probability, decision, and SHAP feature importances for applicant idx."""
    d = get_processed_data()
    if idx < 0 or idx >= len(d["X_test"]):
        raise HTTPException(status_code=404, detail="Applicant index out of bounds")
        
    row = d["X_test"].iloc[idx]
    df_row = pd.DataFrame([row.values], columns=d["feature_names"])
    
    prob_b = float(d["model_b"].predict_proba(df_row)[:, 1][0])
    prob_a = float(d["model_a"].predict_proba(df_row)[:, 1][0])
    
    if prob_b < 0.3:
        decision = "APPROVE"
    elif prob_b <= 0.6:
        decision = "REVIEW"
    else:
        decision = "REJECT"
        
    shap_row = np.atleast_1d(d["shap_vals"][idx])
    
    # Formulate top 5 features by absolute SHAP impact
    feature_impacts = []
    for name, val in zip(d["feature_names"], shap_row):
        feature_impacts.append({
            "feature": name,
            "impact": float(val),
            "abs_impact": float(abs(val))
        })
    feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)
    
    result = {
        "applicant_id": f"APP-{idx:03d}",
        "index": idx,
        "probability_pipeline_b": round(prob_b, 4),
        "probability_pipeline_a": round(prob_a, 4),
        "decision": decision,
        "top_features": feature_impacts[:5],
        "features": {k: float(v) for k, v in row.to_dict().items()}
    }

    # ── Save decision to SQLite ───────────────────────────────────────────────
    conn = get_conn()
    conn.execute(
        "INSERT INTO applicant_decisions (applicant_id, applicant_index, probability, decision) VALUES (?, ?, ?, ?)",
        (f"APP-{idx:03d}", idx, round(prob_b, 4), decision)
    )
    conn.commit()
    conn.close()
    # ─────────────────────────────────────────────────────────────────────────

    return result

@app.get("/api/agents/orchestrate-stream/{idx}")
async def orchestrate_stream(idx: int, request: Request):
    """
    Server-Sent Events (SSE) endpoint that runs the multi-agent pipeline
    and streams each agent's execution stage live to the frontend.
    """
    d = get_processed_data()
    if idx < 0 or idx >= len(d["X_test"]):
        raise HTTPException(status_code=404, detail="Applicant index out of bounds")
        
    row = d["X_test"].iloc[idx]
    df_row = pd.DataFrame([row.values], columns=d["feature_names"])
    prob = float(d["model_b"].predict_proba(df_row)[:, 1][0])
    shap_row = list(np.atleast_1d(d["shap_vals"][idx]))
    feature_names = d["feature_names"]
    decision = "APPROVE" if prob >= 0.5 else "REJECT"
    business_rule = "Standard Algorithmic Review"

    async def event_generator():
        try:
            # Step 1: Start Risk Analyst
            yield {
                "event": "step",
                "data": json.dumps({
                    "agent": "risk_analyst",
                    "status": "running",
                    "message": "Risk Analyst Agent is reviewing ML outputs & SHAP values..."
                })
            }

            await asyncio.sleep(0.2)

            # Run Risk Analyst
            risk_summary = await asyncio.to_thread(
                run_risk_analyst_agent,
                probability=prob,
                shap_values=shap_row,
                feature_names=feature_names,
                business_rule=business_rule
            )

            yield {
                "event": "step",
                "data": json.dumps({
                    "agent": "risk_analyst",
                    "status": "completed",
                    "result": risk_summary
                })
            }

            # Step 2: Start Compliance Check
            yield {
                "event": "step",
                "data": json.dumps({
                    "agent": "compliance",
                    "status": "running",
                    "message": "Compliance Officer Agent is auditing risk summary for fair-lending compliance..."
                })
            }

            await asyncio.sleep(0.2)

            compliance_result = await asyncio.to_thread(
                run_compliance_agent,
                risk_summary=risk_summary
            )

            status = compliance_result.get("status", "ERROR")
            notes = compliance_result.get("notes", "")

            yield {
                "event": "step",
                "data": json.dumps({
                    "agent": "compliance",
                    "status": "completed",
                    "compliance_status": status,
                    "compliance_notes": notes
                })
            }

            # Step 3: Communication Agent
            final_letter = None
            if status == "APPROVED":
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "agent": "communication",
                        "status": "running",
                        "message": "Communication Agent is drafting applicant-facing letter..."
                    })
                }

                await asyncio.sleep(0.2)

                final_letter = await asyncio.to_thread(
                    run_communication_agent,
                    decision=decision,
                    risk_summary=risk_summary
                )

                yield {
                    "event": "step",
                    "data": json.dumps({
                        "agent": "communication",
                        "status": "completed",
                        "final_letter": final_letter
                    })
                }
            else:
                final_letter = "LETTER GENERATION BLOCKED: Compliance flagged this decision for manual human review."
                yield {
                    "event": "step",
                    "data": json.dumps({
                        "agent": "communication",
                        "status": "blocked",
                        "final_letter": final_letter
                    })
                }

            # Save agent run to SQLite
            try:
                conn = get_conn()
                conn.execute(
                    """INSERT INTO agent_runs
                       (applicant_id, risk_summary, compliance_status, compliance_notes, final_letter)
                       VALUES (?, ?, ?, ?, ?)""",
                    (f"APP-{idx:03d}", risk_summary, status, notes, final_letter)
                )
                conn.commit()
                conn.close()
            except Exception as db_err:
                print(f"[DB] Failed to save agent run: {db_err}")

            # Finished
            yield {
                "event": "complete",
                "data": json.dumps({"status": "done"})
            }

        except Exception as e:
            print(f"[SSE] event_generator error: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())

@app.post("/api/agent/underwriter-chat")
async def underwriter_chat(payload: dict):
    """
    Interactive endpoint for underwriter questions with tool support.
    Payload: {"idx": int, "user_query": str}
    """
    idx = payload.get("idx", 0)
    user_query = payload.get("user_query", "")
    
    if not user_query:
        raise HTTPException(status_code=400, detail="user_query is required")
        
    d = get_processed_data()
    idx = max(0, min(idx, len(d["X_test"]) - 1))
    
    row = d["X_test"].iloc[idx]
    df_row = pd.DataFrame([row.values], columns=d["feature_names"])
    prob = float(d["model_b"].predict_proba(df_row)[:, 1][0])
    
    app_context = {
        "applicant_id": f"APP-{idx:03d}",
        "probability": prob,
        "shap_values": list(np.atleast_1d(d["shap_vals"][idx])),
        "feature_names": d["feature_names"],
        "base_data": row.to_dict()
    }
    
    response_text = await asyncio.to_thread(
        run_underwriter_agent,
        user_question=user_query,
        applicant_context=app_context
    )
    
    # ── Save chat to SQLite ───────────────────────────────────────────────────
    conn = get_conn()
    conn.execute(
        "INSERT INTO chat_history (applicant_id, user_query, agent_response) VALUES (?, ?, ?)",
        (f"APP-{idx:03d}", user_query, response_text)
    )
    conn.commit()
    conn.close()
    # ─────────────────────────────────────────────────────────────────────────

    return {
        "applicant_id": f"APP-{idx:03d}",
        "user_query": user_query,
        "agent_response": response_text
    }


# ── History Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/history/decisions")
def get_decision_history(limit: int = 50):
    """Returns recent applicant decisions from the database."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM applicant_decisions ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return {"decisions": [dict(r) for r in rows]}


@app.get("/api/history/agent-runs")
def get_agent_run_history(limit: int = 50):
    """Returns recent agent pipeline runs from the database."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM agent_runs ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return {"agent_runs": [dict(r) for r in rows]}


@app.get("/api/history/chats")
def get_chat_history(applicant_id: str = None, limit: int = 50):
    """Returns underwriter chat history, optionally filtered by applicant."""
    conn = get_conn()
    if applicant_id:
        rows = conn.execute(
            "SELECT * FROM chat_history WHERE applicant_id = ? ORDER BY created_at DESC LIMIT ?",
            (applicant_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_history ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return {"chats": [dict(r) for r in rows]}


@app.get("/api/history/datasets")
def get_dataset_history(limit: int = 20):
    """Returns history of uploaded datasets."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM uploaded_datasets ORDER BY created_at DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return {"datasets": [dict(r) for r in rows]}
