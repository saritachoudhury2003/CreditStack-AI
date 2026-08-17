"""
test_agent_on_applicant.py
--------------------------
End-to-end test script that takes a single Loan_ID, processes it, scores it,
generates SHAP values, and runs it through the multi-agent orchestrator.

Usage: python test_agent_on_applicant.py [Loan_ID]
"""

import sys
import os
import warnings
import pandas as pd
import joblib
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import the orchestrator we built
from agent.orchestrator import run_agent_pipeline

warnings.filterwarnings("ignore")

def main():
    # 1 & 2. Get Loan_ID from args or default to "LP001003"
    target_id = sys.argv[1] if len(sys.argv) > 1 else "LP001003"
    print(f"Testing Agent Pipeline on Applicant: {target_id}")

    # 3. Load full data to find applicant
    data_path = os.path.join("data", "loan_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    if target_id not in df["Loan_ID"].values:
        print(f"Error: Loan_ID '{target_id}' not found in dataset.")
        sys.exit(1)

    # 4. Replicate exact preprocessing steps 
    # (We process the whole df so the StandardScaler fits the exact same distribution)
    df_proc = df.copy()
    
    CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
    NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
    
    for col in CATEGORICAL_COLS:
        df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])
    for col in NUMERIC_COLS:
        df_proc[col] = df_proc[col].fillna(df_proc[col].median())

    df_proc["Gender"]        = df_proc["Gender"].map({"Male": 1, "Female": 0})
    df_proc["Married"]       = df_proc["Married"].map({"Yes": 1, "No": 0})
    df_proc["Education"]     = df_proc["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df_proc["Self_Employed"] = df_proc["Self_Employed"].map({"Yes": 1, "No": 0})
    df_proc["Dependents"]    = df_proc["Dependents"].replace("3+", "3").astype(int)
    
    le = LabelEncoder()
    df_proc["Property_Area"] = le.fit_transform(df_proc["Property_Area"])

    df_proc["Total_Income"]     = df_proc["ApplicantIncome"] + df_proc["CoapplicantIncome"]
    df_proc["Loan_Income_Ratio"] = df_proc["LoanAmount"] / (df_proc["Total_Income"] + 1e-9)

    SCALE_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History", "Total_Income", "Loan_Income_Ratio"]
    scaler = StandardScaler()
    df_proc[SCALE_COLS] = scaler.fit_transform(df_proc[SCALE_COLS])

    # Extract the single processed row for our target applicant
    applicant_row = df_proc[df_proc["Loan_ID"] == target_id]
    
    # Drop non-feature columns
    X_applicant = applicant_row.drop(columns=["Loan_ID", "Loan_Status"])
    feature_names = X_applicant.columns.tolist()

    # 5. Load model and predict
    print("\nLoading model and scoring...")
    model_path = os.path.join("models", "pipeline_b_model.pkl")
    model = joblib.load(model_path)
    
    prob = float(model.predict_proba(X_applicant)[:, 1][0])

    # 6. Compute SHAP values for this specific applicant
    print("Computing SHAP values (may take ~10 seconds)...")
    # Load original X_test for background kmeans distribution
    X_test_path = os.path.join("processed_data", "X_test.pkl")
    X_test = joblib.load(X_test_path)
    background = shap.kmeans(X_test.values, 10)
    
    def predict_fn(X_arr):
        tmp_df = pd.DataFrame(X_arr, columns=feature_names)
        return model.predict_proba(tmp_df)[:, 1]
        
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_vals = explainer.shap_values(X_applicant.values, nsamples=100, silent=True)[0]

    # 7. Determine decision using thresholds
    if prob < 0.3:
        decision = "APPROVE"
    elif prob <= 0.6:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    # 8. Call orchestrator
    print("\n========================================================")
    print("      SCORING & AGENT ORCHESTRATOR PIPELINE START       ")
    print("========================================================")
    print(f"Applicant ID : {target_id}")
    print(f"Probability  : {prob:.2%}")
    print(f"System Action: {decision} (Threshold logic)")
    
    print("\n>> Invoking Agent Swarm...")
    result = run_agent_pipeline(
        probability=prob,
        shap_values=list(shap_vals),
        feature_names=feature_names,
        business_rule="Standard algorithmic review",
        decision=decision
    )
    
    # 9. Print results clearly formatted
    print("\n--------------------------------------------------------")
    print("[1] RISK ANALYST SUMMARY")
    print("--------------------------------------------------------")
    print(result.get("risk_summary", "ERROR"))
    
    print("\n--------------------------------------------------------")
    print("[2] COMPLIANCE CHECK")
    print("--------------------------------------------------------")
    print(f"Status: {result.get('compliance_status', 'ERROR')}")
    print(f"Notes : {result.get('compliance_notes', 'ERROR')}")
    
    print("\n--------------------------------------------------------")
    print("[3] COMMUNICATION AGENT LETTER")
    print("--------------------------------------------------------")
    print(result.get("final_letter", "ERROR"))
    print("\n========================================================")
    print("                       DONE                             ")
    print("========================================================")

if __name__ == "__main__":
    main()
