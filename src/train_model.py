# =============================================
# src/train_model.py
# =============================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, silhouette_score
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


LEAK_EXACT = [
    "ChurnRisk", "AccountStatus", "CustomerType", "RFMSegment",
    "Satisfaction", "SupportTickets", "LoyaltyLevel", "SpendingCat",
    "CustomerID", "Churn", "Recency", "FavoriteSeason",
    "CustomerTenureDays", "PreferredMonth"
]

LEAK_KEYWORDS = [
    "churn", "risk", "accountstatus", "customertype", "perdu", "closed",
    "loyaltylevel", "rfmsegment", "spendingcat", "satisfaction",
    "supportticket", "customerid", "recency", "favoriteseason",
    "customertenuredays", "preferredmonth"
]

ORDINAL_CANDIDATES = [
    "AgeCategory", "BasketSizeCategory", "PreferredTimeOfDay",
    "ProductDiversity", "WeekendPreference"
]


# ─────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────

def load_splits():
    """Read the train/test splits produced by preprocessing."""
    X_train = pd.read_csv("data/train_test/X_train.csv")
    X_test  = pd.read_csv("data/train_test/X_test.csv")
    y_train = pd.read_csv("data/train_test/y_train.csv").squeeze()
    y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()
    print(
        f"[OK] Splits loaded — {X_train.shape[1]} features | "
        f"{X_train.shape[0]} train samples | {X_test.shape[0]} test samples\n"
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# Leakage guard
# ─────────────────────────────────────────────

def strip_leaking_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Remove any remaining columns that could leak the Churn target."""
    flagged = {
        col for col in X_train.columns
        if col in LEAK_EXACT or any(kw in col.lower() for kw in LEAK_KEYWORDS)
    }
    if flagged:
        print(f"[LEAK] {len(flagged)} leaking features removed: {sorted(flagged)}")
        X_train = X_train.drop(columns=[c for c in flagged if c in X_train.columns])
        X_test  = X_test.drop( columns=[c for c in flagged if c in X_test.columns])
    else:
        print("[LEAK] No leaking features detected.")
    print(f"       Remaining features: {X_train.shape[1]}\n")
    return X_train, X_test


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────

def _build_column_transformer(X_train: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for categorical features."""
    cat_cols     = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    ordinal_cols = [c for c in cat_cols if c in ORDINAL_CANDIDATES]
    onehot_cols  = [c for c in cat_cols if c not in ordinal_cols]

    transformers = []
    if onehot_cols:
        transformers.append((
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            onehot_cols
        ))
    if ordinal_cols:
        transformers.append((
            "ordinal",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ordinal_cols
        ))

    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Fit encoder on train, transform both splits; save encoder to disk."""
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_cols:
        print("[OK] No categorical features to encode.\n")
        return X_train, X_test, None

    print(f"[ENC] Encoding {len(cat_cols)} categorical columns: {cat_cols}")

    transformer = _build_column_transformer(X_train)
    transformer.fit(X_train)

    feature_names = transformer.get_feature_names_out()

    X_train_enc = pd.DataFrame(
        transformer.transform(X_train),
        columns=feature_names,
        index=X_train.index
    )
    X_test_enc = pd.DataFrame(
        transformer.transform(X_test),
        columns=feature_names,
        index=X_test.index
    )

    # Second leakage pass on encoded column names
    leaked_enc = [c for c in X_train_enc.columns
                  if any(kw in c.lower() for kw in LEAK_KEYWORDS)]
    if leaked_enc:
        print(f"[LEAK] Encoded leaking columns removed: {leaked_enc}")
        X_train_enc.drop(columns=leaked_enc, inplace=True)
        X_test_enc.drop(columns=[c for c in leaked_enc if c in X_test_enc.columns], inplace=True)

    print(f"[OK] Encoding done — {X_train_enc.shape[1]} columns\n")

    os.makedirs("models", exist_ok=True)
    joblib.dump(transformer, "models/encoder.pkl")
    print("[SAVED] models/encoder.pkl\n")

    return X_train_enc, X_test_enc, transformer


# ─────────────────────────────────────────────
# Diagnostic
# ─────────────────────────────────────────────

def leakage_diagnostic(X_train: pd.DataFrame, y_train: pd.Series) -> None:
    """Print top 15 feature correlations with the target."""
    print("[DIAG] Top 15 feature correlations with Churn:")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    corr_series = X_train[num_cols].corrwith(y_train).abs().sort_values(ascending=False)
    print(corr_series.head(15).to_string())

    peak = corr_series.iloc[0] if len(corr_series) > 0 else 0
    if peak > 0.7:
        print(f"\n[WARN] Leakage suspected: {corr_series.index[0]} (corr={peak:.3f})")
    else:
        print(f"\n[OK]   Max correlation = {peak:.3f} — no obvious leakage")
    print("-" * 60 + "\n")


# ─────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────

def fit_kmeans(X_train: pd.DataFrame, n_clusters: int = 4) -> KMeans:
    """Fit KMeans on numerical features and save the model."""
    print(f"[CLUSTER] KMeans with {n_clusters} clusters...")
    num_data = X_train.select_dtypes(include=[np.number])

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(num_data)

    sil = silhouette_score(num_data, labels) if num_data.shape[1] >= 2 else 0.0
    print(f"[OK] Silhouette score: {sil:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/kmeans_model.pkl")
    print("[SAVED] models/kmeans_model.pkl\n")
    return model


# ─────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────

def fit_random_forest(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series
) -> RandomForestClassifier:
    """Grid-search a RandomForest classifier and save results."""
    print("[RF] Training RandomForestClassifier...")

    param_grid = {
        "n_estimators":      [100, 200],
        "max_depth":         [6, 10, 15],
        "min_samples_split": [5, 10],
        "min_samples_leaf":  [2, 5],
    }

    base_rf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    search  = GridSearchCV(base_rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print(f"[RF] Best params: {search.best_params_}\n")
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_test, predictions, digits=3))
    print(f"  Accuracy : {accuracy_score(y_test, predictions):.4f}")
    print(f"  F1-Score : {f1_score(y_test, predictions):.4f}\n")

    os.makedirs("reports", exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"], ax=ax)
    ax.set_title("Confusion Matrix — Churn Prediction")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    fig.tight_layout()
    fig.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Feature importance
    importance = pd.Series(
        best_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.head(20).plot(kind="bar", ax=ax)
    ax.set_title("Top 20 Feature Importances — RandomForest")
    fig.tight_layout()
    fig.savefig("reports/feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("[SAVED] reports/confusion_matrix.png")
    print("[SAVED] reports/feature_importance.png\n")

    joblib.dump(best_model, "models/randomforest_churn.pkl")
    print("[SAVED] models/randomforest_churn.pkl\n")
    return best_model


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_training():
    X_train, X_test, y_train, y_test = load_splits()

    X_train, X_test = strip_leaking_features(X_train, X_test)

    # Clustering on raw numerical data (before encoding)
    fit_kmeans(X_train)

    # Encode categoricals and save encoder for Flask
    X_train_enc, X_test_enc, _ = encode_categoricals(X_train, X_test)

    # Post-encoding leakage check
    leakage_diagnostic(X_train_enc, y_train)

    # Train classifier
    fit_random_forest(X_train_enc, X_test_enc, y_train, y_test)

    print("=" * 60)
    print("  Training complete.")
    print("  models/ : randomforest_churn.pkl, kmeans_model.pkl, encoder.pkl")
    print("  reports/: confusion_matrix.png, feature_importance.png")
    print("=" * 60)


if __name__ == "__main__":
    run_training()