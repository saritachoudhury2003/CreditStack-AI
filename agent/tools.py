"""
tools.py
--------
Functions defining tools that the interactive Underwriter Agent can use.
"""

import os
import joblib
import pandas as pd
import numpy as np

def simulate_alternate_scenario(new_income: float, new_dti: float, base_applicant_data: dict) -> float:
    """
    Takes modified income and DTI ratios, re-runs them through Pipeline B,
    and returns the new predicted probability.
    """
    # Load the trained model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "..", "models", "pipeline_b_model.pkl")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        return f"Error loading model: {e}"
        
    # Reconstruct a DataFrame for a single prediction
    # Assuming base_applicant_data contains the exact features expected by the model
    # (after preprocessing)
    df = pd.DataFrame([base_applicant_data])
    
    # Update the requested fields (mapping DTI to our feature name Loan_Income_Ratio)
    if "Total_Income" in df.columns:
        df["Total_Income"] = new_income
    if "Loan_Income_Ratio" in df.columns:
        df["Loan_Income_Ratio"] = new_dti
        
    # Re-predict
    try:
        prob = model.predict_proba(df)[:, 1][0]
        return round(prob, 4)
    except Exception as e:
        return f"Error during prediction: {e}"

def fetch_applicant_docs(applicant_id: str) -> dict:
    """
    Mock function returning fake document status.
    """
    # In a real system, this would query a database
    return {
        "applicant_id": applicant_id,
        "income_proof": "verified",
        "id_proof": "pending",
        "address_proof": "verified"
    }

def get_shap_explanation(applicant_id: str, shap_values: list, feature_names: list) -> str:
    """
    Takes existing SHAP values and feature names, returning a formatted string
    listing the top 3 contributing features and their impact.
    """
    impacts = list(zip(feature_names, shap_values))
    # Sort by absolute impact
    impacts.sort(key=lambda x: abs(x[1]), reverse=True)
    top_3 = impacts[:3]
    
    explanation = f"Top 3 risk drivers for applicant {applicant_id}:\n"
    for i, (name, val) in enumerate(top_3, 1):
        direction = "Positive (Improves odds)" if val > 0 else "Negative (Reduces odds)"
        explanation += f"{i}. {name}: {val:.4f} impact [{direction}]\n"
        
    return explanation
