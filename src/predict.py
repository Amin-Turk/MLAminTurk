# =============================================
# src/predict.py
# =============================================

import os
import joblib
import numpy as np
import pandas as pd


def load_artifacts():
    """Load all trained models and the encoder from disk."""
    kmeans    = joblib.load("models/kmeans_model.pkl")
    rf        = joblib.load("models/randomforest_churn.pkl")
    encoder   = joblib.load("models/encoder.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    return kmeans, rf, encoder, scaler


def prepare_input(client: dict, encoder, scaler) -> pd.DataFrame:
    """Convert a raw client dict into a model-ready DataFrame."""
    df = pd.DataFrame([client])

    # Scale numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = scaler.transform(df[num_cols])

    # Encode categoricals
    cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
    if cat_cols:
        encoded = pd.DataFrame(
            encoder.transform(df),
            columns=encoder.get_feature_names_out(),
            index=df.index
        )
        df = encoded

    return df


def predict_client(client: dict) -> dict:
    """Predict churn risk and customer segment for a single client."""
    kmeans, rf, encoder, scaler = load_artifacts()

    df = prepare_input(client, encoder, scaler)

    churn_pred  = rf.predict(df)[0]
    churn_proba = rf.predict_proba(df)[0][1]
    segment     = kmeans.predict(df.select_dtypes(include=[np.number]))[0]

    recommendation = (
        "Launch retention campaign"
        if churn_pred == 0
        else "Send recovery offer immediately"
    )

    return {
        "churn_prediction" : "Churned" if churn_pred == 1 else "Retained",
        "churn_probability": round(float(churn_proba), 4),
        "customer_segment" : int(segment),
        "recommendation"   : recommendation,
    }


if __name__ == "__main__":
    sample_client = {
        "Frequency"      : 12,
        "MonetaryTotal"  : 1250.75,
        "CustomerTenure" : 180,
        # add remaining features to match training columns
    }

    result = predict_client(sample_client)
    print("\n── Prediction Result ──────────────────────")
    for key, val in result.items():
        print(f"  {key:<22}: {val}")
    print("────────────────────────────────────────────")