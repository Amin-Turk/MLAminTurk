# =============================================
# src/preprocessing.py
# =============================================

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import (
    read_dataset, export_dataset, correlation_heatmap,
    drop_collinear_features, build_features,
    drop_low_info_features, fill_missing_values, run_pca
)


# ─────────────────────────────────────────────
# Columns that leak the Churn target
# (derived from Churn or post-event features)
# ─────────────────────────────────────────────
LEAKING_COLS = [
    "ChurnRisk", "AccountStatus", "CustomerType", "RFMSegment",
    "Satisfaction", "SupportTickets", "LoyaltyLevel",
    "SpendingCat", "CustomerID", "Churn",
    "Recency",
    "FavoriteSeason",
    "CustomerTenureDays",
    "PreferredMonth",
]

LEAK_KEYWORDS = [
    "churn", "risk", "accountstatus", "customertype",
    "perdu", "closed", "loyaltylevel", "rfmsegment",
    "spendingcat", "satisfaction", "supportticket"
]


def remove_leaking_cols(X: pd.DataFrame) -> pd.DataFrame:
    """Drop all columns that could leak the target variable."""
    cols_to_drop = set()

    # Exact matches
    for col in LEAKING_COLS:
        if col in X.columns:
            cols_to_drop.add(col)

    # Encoded variants (e.g. OneHot creates "AccountStatus_Closed")
    for col in X.columns:
        if any(kw in col.lower() for kw in LEAK_KEYWORDS):
            cols_to_drop.add(col)

    if cols_to_drop:
        print(f"[LEAK] {len(cols_to_drop)} leaking columns removed: {sorted(cols_to_drop)}")
        X = X.drop(columns=list(cols_to_drop))
    else:
        print("[LEAK] No leaking columns detected.")

    return X


def run_preprocessing():
    print("=" * 60)
    print("  Preprocessing pipeline — anti-leakage")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────────
    df = read_dataset("data/raw/retail.csv")

    # ── 2. Feature engineering & parsing ─────────────────────────
    print("\n[STEP 2] Feature engineering + date/IP parsing...")
    df = build_features(df)

    # ── 3. Split target from features ────────────────────────────
    target = "Churn"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    y = df[target].copy()
    X = df.drop(columns=[target])

    # ── 4. Remove leaking features ────────────────────────────────
    print("\n[STEP 4] Removing leaking features...")
    X = remove_leaking_cols(X)

    # ── 5. Remove low-information features ───────────────────────
    print("\n[STEP 5] Dropping zero-variance and high-NaN columns...")
    X = drop_low_info_features(X)

    # ── 6. Train / test split ─────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[SPLIT] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"        Churn distribution (train): {y_train.value_counts().to_dict()}")

    # Keep test columns aligned with train
    X_test = X_test[X_train.columns]

    # ── 7. Impute missing values ──────────────────────────────────
    print("\n[STEP 7] Imputing missing values...")
    X_train, X_test = fill_missing_values(X_train, X_test)

    # ── 8. Correlation & multicollinearity ───────────────────────
    print("\n[STEP 8] Correlation analysis + dropping collinear features...")
    _, flagged = correlation_heatmap(X_train)
    X_train = drop_collinear_features(X_train, threshold=0.85)
    X_test  = X_test[[c for c in X_train.columns if c in X_test.columns]]

    # ── 9. Leakage diagnostic ─────────────────────────────────────
    print("\n[DIAG] Top 15 correlations with Churn (numerical features):")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    churn_corr = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)
    print(churn_corr.head(15).to_string())

    top_corr = churn_corr.iloc[0] if len(churn_corr) > 0 else 0
    if top_corr > 0.7:
        print(f"\n[WARN] Max correlation = {top_corr:.3f} — potential leakage!")
        print(f"       Suspect feature: {churn_corr.index[0]}")
    else:
        print(f"\n[OK]   Max correlation = {top_corr:.3f} — no obvious leakage")
    print("-" * 60)

    # ── 10. Scaling ───────────────────────────────────────────────
    print("\n[STEP 10] Applying StandardScaler (fit on train only)...")
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[num_features] = scaler.fit_transform(X_train[num_features])
    X_test_scaled[num_features]  = scaler.transform(X_test[num_features])

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("[SAVED] models/scaler.pkl")

    # ── 11. PCA ───────────────────────────────────────────────────
    print("\n[STEP 11] Running PCA on scaled numerical features...")
    X_train_pca, X_test_pca, pca_model = run_pca(
        X_train_scaled[num_features],
        X_test_scaled[num_features]
    )

    # ── 12. Save outputs ──────────────────────────────────────────
    for folder in ["data/processed", "data/train_test", "reports"]:
        os.makedirs(folder, exist_ok=True)

    # Raw splits for train_model.py
    X_train.to_csv("data/train_test/X_train.csv", index=False)
    X_test.to_csv("data/train_test/X_test.csv",   index=False)
    y_train.to_csv("data/train_test/y_train.csv",  index=False, header=True)
    y_test.to_csv("data/train_test/y_test.csv",    index=False, header=True)

    # PCA versions
    export_dataset(X_train_pca, "data/processed/X_train_pca.csv")
    export_dataset(X_test_pca,  "data/processed/X_test_pca.csv")

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Final feature count : {X_train.shape[1]}")
    print("  data/train_test/    : ready for train_model.py")
    print("  data/processed/     : PCA versions available")
    print("=" * 60)

    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca_model


if __name__ == "__main__":
    run_preprocessing()