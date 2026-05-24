# =============================================
# app/app.py  — Flask Deployment Interface
# =============================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__, template_folder="../templates")


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

MODEL_PATHS = {
    "rf"     : "models/randomforest_churn.pkl",
    "kmeans" : "models/kmeans_model.pkl",
    "scaler" : "models/scaler.pkl",
    "encoder": "models/encoder.pkl",
}

def load_models() -> dict:
    models = {}
    for key, path in MODEL_PATHS.items():
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            print(f"[WARN] {path} not found — some features will be unavailable")
    return models

MODELS = load_models()

try:
    EXPECTED_FEATURES = MODELS["rf"].feature_names_in_.tolist()
except Exception:
    EXPECTED_FEATURES = []

LEAK_KEYWORDS = [
    "churn", "risk", "accountstatus", "customertype", "perdu", "closed",
    "loyaltylevel", "rfmsegment", "spendingcat", "satisfaction",
    "supportticket", "customerid", "recency", "favoriteseason",
    "customertenuredays", "preferredmonth"
]

CLUSTER_LABELS = {
    0: ("Champions",  "Frequent buyers, high value",          "#22c55e"),
    1: ("Loyal",      "Regular customers, good retention",    "#3b82f6"),
    2: ("Potential",  "Active but not yet consistent",        "#f59e0b"),
    3: ("Dormant",    "Inactive, high churn risk",            "#ef4444"),
}


# ─────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────

def apply_encoder(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns using the saved ColumnTransformer."""
    enc = MODELS.get("encoder")
    if enc is not None:
        try:
            return pd.DataFrame(
                enc.transform(df),
                columns=enc.get_feature_names_out(),
                index=df.index
            )
        except Exception as e:
            print(f"[WARN] Encoder failed ({e}), falling back to get_dummies")

    # Fallback
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols)
    return df


def align_to_model(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing columns and reorder to match training feature set."""
    if EXPECTED_FEATURES:
        for col in EXPECTED_FEATURES:
            if col not in df.columns:
                df[col] = 0
        df = df[EXPECTED_FEATURES]
    return df.fillna(0)


def prepare_single(raw: dict) -> pd.DataFrame:
    """Convert a raw input dict into a model-ready DataFrame."""
    df = pd.DataFrame([raw])

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    leak_cols = [c for c in df.columns if any(kw in c.lower() for kw in LEAK_KEYWORDS)]
    df.drop(columns=leak_cols, errors="ignore", inplace=True)

    df = apply_encoder(df)
    df = align_to_model(df)
    return df


def prepare_batch(X: pd.DataFrame) -> pd.DataFrame:
    """Encode and align a full DataFrame (e.g. X_test.csv) for batch prediction."""
    leak_cols = [c for c in X.columns if any(kw in c.lower() for kw in LEAK_KEYWORDS)]
    X = X.drop(columns=leak_cols, errors="ignore")
    X = apply_encoder(X)
    X = align_to_model(X)
    return X


def get_risk_label(proba: float):
    if proba >= 0.75: return ("Critical", "#ef4444")
    if proba >= 0.50: return ("High",     "#f97316")
    if proba >= 0.25: return ("Medium",   "#f59e0b")
    return                    ("Low",      "#22c55e")


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("predict.html", result=None)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", result=None)

    try:
        form_data = request.form.to_dict()
        df = prepare_single(form_data)

        rf = MODELS.get("rf")
        if rf is None:
            return render_template("predict.html",
                                   error="RandomForest model not loaded. Run src/train_model.py first.",
                                   result=None)

        pred  = int(rf.predict(df)[0])
        proba = float(rf.predict_proba(df)[0][1])
        risk, risk_color = get_risk_label(proba)

        result = {
            "prediction" : pred,
            "label"      : "Churned" if pred == 1 else "Retained",
            "probability": round(proba * 100, 1),
            "risk"       : risk,
            "risk_color" : risk_color,
        }
        return render_template("predict.html", result=result, form=form_data)

    except Exception as e:
        return render_template("predict.html", error=str(e), result=None)
if __name__ == "__main__":
    print("[OK] Starting Flask app → http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)