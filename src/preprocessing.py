"""
preprocessing.py
----------------
Core preprocessing logic wrapped in preprocess_data(df).
Runnable standalone via __main__ using data/loan_data.csv.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


def preprocess_data(df: pd.DataFrame):
    """
    Takes a raw loan DataFrame and returns processed train/test splits.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loan data matching the loan_data.csv schema.

    Returns
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    """
    df = df.copy()

    # ── Drop identifier column if present ──────────────────────────
    if "Loan_ID" in df.columns:
        df.drop(columns=["Loan_ID"], inplace=True)

    # ── Validate required columns ───────────────────────────────────
    required = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History", "Property_Area", "Loan_Status",
    ]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # ── Impute missing values ───────────────────────────────────────
    CATEGORICAL_COLS = [
        "Gender", "Married", "Dependents",
        "Education", "Self_Employed", "Property_Area",
    ]
    NUMERIC_COLS = [
        "ApplicantIncome", "CoapplicantIncome",
        "LoanAmount", "Loan_Amount_Term", "Credit_History",
    ]

    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    # ── Encode categorical features ─────────────────────────────────
    df["Gender"]        = df["Gender"].map({"Male": 1, "Female": 0})
    df["Married"]       = df["Married"].map({"Yes": 1, "No": 0})
    df["Education"]     = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    df["Dependents"]    = df["Dependents"].replace("3+", "3").astype(int)

    le = LabelEncoder()
    df["Property_Area"] = le.fit_transform(df["Property_Area"])

    # ── Encode target ───────────────────────────────────────────────
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    # ── Feature engineering ─────────────────────────────────────────
    df["Total_Income"]     = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["Total_Income"] + 1e-9)

    # ── Scale numeric features ──────────────────────────────────────
    SCALE_COLS = [
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
        "Loan_Amount_Term", "Credit_History",
        "Total_Income", "Loan_Income_Ratio",
    ]
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]

    scaler = StandardScaler()
    X[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])

    # ── Train / test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


# ───────────────────────────────────────────────────────────────────
# Standalone execution — saves processed splits to processed_data/
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH     = os.path.join(BASE_DIR, "..", "data", "loan_data.csv")
    OUT_DIR       = os.path.join(BASE_DIR, "..", "processed_data")

    print("Loading data/loan_data.csv ...")
    raw_df = pd.read_csv(DATA_PATH)
    print(f"  Raw shape      : {raw_df.shape}")
    print(f"  Missing values : {raw_df.isnull().sum().sum()}")

    X_train, X_test, y_train, y_test = preprocess_data(raw_df)

    print(f"\nAfter preprocessing:")
    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}  |  y_test  : {y_test.shape}")
    print(f"  Features: {X_train.columns.tolist()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(X_train, os.path.join(OUT_DIR, "X_train.pkl"))
    joblib.dump(X_test,  os.path.join(OUT_DIR, "X_test.pkl"))
    joblib.dump(y_train, os.path.join(OUT_DIR, "y_train.pkl"))
    joblib.dump(y_test,  os.path.join(OUT_DIR, "y_test.pkl"))
    print(f"\n[DONE] Splits saved to {os.path.abspath(OUT_DIR)}")
