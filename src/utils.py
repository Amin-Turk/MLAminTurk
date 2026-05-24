import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from typing import Tuple, List


# ─────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────

def read_dataset(path: str = "data/raw/retail.csv") -> pd.DataFrame:
    """Load the raw CSV file into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Make sure your CSV is placed inside data/raw/"
        )
    dataset = pd.read_csv(path)
    print(f"[OK] Dataset loaded — {dataset.shape[0]:,} rows × {dataset.shape[1]} columns")
    return dataset


def export_dataset(dataset: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV at the given path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset.to_csv(path, index=False)
    print(f"[SAVED] {path}")


# ─────────────────────────────────────────────
# Correlation analysis
# ─────────────────────────────────────────────

def correlation_heatmap(
    dataset: pd.DataFrame,
    threshold: float = 0.8,
    output_path: str = "reports/correlation_heatmap.png"
) -> Tuple[pd.DataFrame, List[tuple]]:
    """Plot and save a correlation heatmap; return the matrix and highly correlated pairs."""
    num_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    corr = dataset[num_features].corr()

    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0,
                linewidths=0.4, square=True, ax=ax)
    ax.set_title("Correlation Matrix — Numerical Features", fontsize=16)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Heatmap saved → {output_path}")

    # Identify highly correlated pairs
    flagged_pairs = []
    n = len(corr.columns)
    for i in range(n):
        for j in range(i + 1, n):
            val = corr.iloc[i, j]
            if abs(val) > threshold:
                flagged_pairs.append((corr.columns[i], corr.columns[j], round(val, 3)))

    if flagged_pairs:
        print(f"[WARN] {len(flagged_pairs)} highly correlated pairs found (|r| > {threshold})")

    return corr, flagged_pairs


def drop_collinear_features(dataset: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Remove features causing multicollinearity above the given threshold."""
    num_features = dataset.select_dtypes(include=[np.number]).columns.tolist()
    abs_corr = dataset[num_features].corr().abs()
    upper_tri = abs_corr.where(
        np.triu(np.ones(abs_corr.shape), k=1).astype(bool)
    )
    cols_to_remove = [
        col for col in upper_tri.columns if any(upper_tri[col] > threshold)
    ]
    if cols_to_remove:
        print(f"[DROP] Collinear features removed: {cols_to_remove}")
        dataset = dataset.drop(columns=cols_to_remove)
    return dataset


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────

def build_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create derived features and parse raw columns (dates, IP addresses)."""
    df = dataset.copy()

    # Business ratios
    if {"MonetaryTotal", "Recency"}.issubset(df.columns):
        df["SpendPerDay"] = df["MonetaryTotal"] / (df["Recency"] + 1)
    if {"MonetaryTotal", "Frequency"}.issubset(df.columns):
        df["BasketAvg"] = df["MonetaryTotal"] / df["Frequency"]
    if {"Recency", "CustomerTenure"}.issubset(df.columns):
        df["RecencyRatio"] = df["Recency"] / df["CustomerTenure"].replace(0, np.nan)

    # Parse registration date
    date_col = next(
        (c for c in ["RegistrationDate", "RegistDate"] if c in df.columns), None
    )
    if date_col:
        parsed_dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
        df["RegYear"]    = parsed_dates.dt.year
        df["RegMonth"]   = parsed_dates.dt.month
        df["RegDay"]     = parsed_dates.dt.day
        df["RegWeekday"] = parsed_dates.dt.weekday
        df.drop(columns=[date_col], inplace=True)
        print(f"[OK] '{date_col}' parsed → RegYear, RegMonth, RegDay, RegWeekday")

    # Parse LastLoginIP
    if "LastLoginIP" in df.columns:
        def _parse_ip(ip_val):
            if pd.isna(ip_val):
                return pd.Series({"PrivateIP": np.nan, "IPVer": np.nan})
            raw = str(ip_val).strip()
            is_private = int(bool(
                re.match(r"^(10\.|172\.(1[6-9]|2[0-9]|3[01])\.|192\.168\.)", raw)
            ))
            if re.match(r"^\d+\.\d+\.\d+\.\d+$", raw):
                version = 4
            elif ":" in raw:
                version = 6
            else:
                version = 0
            return pd.Series({"PrivateIP": is_private, "IPVer": version})

        ip_parsed = df["LastLoginIP"].apply(_parse_ip)
        df = pd.concat([df.drop(columns=["LastLoginIP"]), ip_parsed], axis=1)
        print("[OK] 'LastLoginIP' parsed → PrivateIP, IPVer")

    return df


# ─────────────────────────────────────────────
# Feature cleaning
# ─────────────────────────────────────────────

def drop_low_info_features(dataset: pd.DataFrame, nan_threshold: float = 0.5) -> pd.DataFrame:
    """Remove constant columns and columns with too many missing values."""
    df = dataset.copy()

    # Constant or near-constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].var(ddof=0) == 0:
            constant_cols.append(col)

    if constant_cols:
        print(f"[DROP] Constant/zero-variance columns: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)

    # Too many NaN
    missing_rate = df.isnull().mean()
    high_nan_cols = missing_rate[missing_rate > nan_threshold].index.tolist()
    if high_nan_cols:
        print(f"[DROP] Columns >{nan_threshold*100:.0f}% missing: {high_nan_cols}")
        df.drop(columns=high_nan_cols, inplace=True)

    return df


# ─────────────────────────────────────────────
# Imputation
# ─────────────────────────────────────────────

def fill_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values: KNN for numerics, mode for categoricals."""
    train = X_train.copy()
    test  = X_test.copy()

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train.select_dtypes(include=["object", "category"]).columns.tolist()

    if num_cols:
        knn_imp = KNNImputer(n_neighbors=5)
        train[num_cols] = knn_imp.fit_transform(train[num_cols])
        test[num_cols]  = knn_imp.transform(test[num_cols])
        print(f"[OK] KNN imputation on {len(num_cols)} numerical features")

    if cat_cols:
        mode_imp = SimpleImputer(strategy="most_frequent")
        train[cat_cols] = mode_imp.fit_transform(train[cat_cols])
        test[cat_cols]  = mode_imp.transform(test[cat_cols])
        print(f"[OK] Mode imputation on {len(cat_cols)} categorical features")

    return train, test


# ─────────────────────────────────────────────
# Dimensionality reduction
# ─────────────────────────────────────────────

def run_pca(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    n_components: int = None,
    variance_target: float = 0.95,
    model_path: str = "models/pca_model.pkl"
) -> Tuple[pd.DataFrame, pd.DataFrame, PCA]:
    """Fit PCA on training data, transform both splits, and save the model."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Auto-select number of components
    if n_components is None:
        probe = PCA().fit(X_train)
        cumulative_var = np.cumsum(probe.explained_variance_ratio_)
        n_components = int(np.argmax(cumulative_var >= variance_target) + 1)
        print(f"[PCA] {n_components} components selected ({variance_target*100:.0f}% variance retained)")

    pca_model = PCA(n_components=n_components, random_state=42)
    train_pca = pca_model.fit_transform(X_train)
    test_pca  = pca_model.transform(X_test)

    joblib.dump(pca_model, model_path)
    print(f"[SAVED] PCA model → {model_path}")

    # Scree plot
    cumvar = np.cumsum(pca_model.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, n_components + 1), cumvar, marker="o", color="steelblue")
    ax.axhline(y=variance_target, color="crimson", linestyle="--", label=f"{variance_target*100:.0f}% threshold")
    ax.set_title("Cumulative Explained Variance — PCA")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance")
    ax.legend()
    ax.grid(True, alpha=0.4)
    fig.savefig("reports/pca_variance_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    component_cols = [f"PC{i + 1}" for i in range(n_components)]
    return (
        pd.DataFrame(train_pca, columns=component_cols, index=X_train.index),
        pd.DataFrame(test_pca,  columns=component_cols, index=X_test.index),
        pca_model,
    )