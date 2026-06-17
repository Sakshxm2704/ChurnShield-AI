"""
ml/features/preprocessing.py
-----------------------------
Data preprocessing pipeline for the Telco Churn dataset.

Steps
-----
1. Load raw CSV
2. Drop duplicates
3. Handle missing values (TotalCharges is the main culprit)
4. Type casting
5. Binary-encode Yes/No and gender columns
6. One-hot encode nominal categoricals
7. Scale numeric features (StandardScaler)
8. Train/test split
9. Persist the fitted preprocessor (sklearn Pipeline) via joblib

Public API
----------
- ``load_raw_data(path)``         → raw DataFrame
- ``clean_data(df)``              → cleaned DataFrame
- ``build_preprocessor(df)``      → fitted ColumnTransformer
- ``preprocess(df, preprocessor)``→ (X_array, feature_names)
- ``run_preprocessing_pipeline()``→ (X_train, X_test, y_train, y_test, preprocessor, feature_names)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.config import (
    BINARY_COLS,
    FEATURE_NAMES_FILENAME,
    MODEL_DIR,
    NOMINAL_COLS,
    NUMERIC_COLS,
    PREPROCESSOR_FILENAME,
    PROC_DATASET_PATH,
    RAW_DATASET_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    ID_COL,
)

logger = logging.getLogger(__name__)


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_raw_data(path: Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load CSV from *path* and return a raw DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run: python -m data.generate_dataset"
        )
    df = pd.read_csv(path, dtype=str)   # load everything as str first
    logger.info("Loaded raw data: %d rows × %d cols", *df.shape)
    return df


# ── 2. Clean ───────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned copy of *df*.

    Operations
    ----------
    - Strip whitespace from string columns
    - Drop exact duplicate rows
    - Drop the ID column (not predictive)
    - Convert TotalCharges to numeric (coerce empty strings → NaN)
    - Fill TotalCharges NaN with median
    - Cast numeric columns
    - Cast target to int
    """
    df = df.copy()

    # Strip whitespace
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d duplicate rows.", dropped)

    # Drop ID column
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])



    # TotalCharges: absent in inference payloads → default 0
    if "TotalCharges" not in df.columns:
        df["TotalCharges"] = 0.0
    # TotalCharges: empty string -> NaN -> median fill
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        df["TotalCharges"] = df["TotalCharges"].fillna(median_tc)
        logger.info("Imputed %d missing TotalCharges with median %.2f.", n_missing, median_tc)

    # Cast numeric columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # SeniorCitizen is already 0/1 string → int
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    # Target column
    if TARGET_COL in df.columns:
        # Support both numeric (0/1) and string ("Yes"/"No") formats
        if df[TARGET_COL].dtype == object:
            df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)
        else:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)

    logger.info("Cleaned data: %d rows × %d cols", *df.shape)
    return df


# ── 3. Binary encoding (Yes/No + gender) ──────────────────────────────────────

def _binary_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    In-place binary encoding for Yes/No columns and gender.
    Columns not present in the DataFrame are silently skipped.
    """
    df = df.copy()
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        if col == "gender":
            # Already numeric → skip
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = (df[col].astype(str).str.lower() == "female").astype(int)
        else:
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    return df


# ── 4. Build sklearn ColumnTransformer ────────────────────────────────────────

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Fit and return a ``ColumnTransformer`` on the cleaned DataFrame *df*.

    Transformations
    ---------------
    - Numeric cols : median impute → StandardScaler
    - Nominal cols : constant impute ("Missing") → OneHotEncoder (drop first)
    """
    # Only include columns that actually exist in df
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    nom_cols = [c for c in NOMINAL_COLS if c in df.columns]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("nom", nominal_pipeline, nom_cols),
        ],
        remainder="drop",   # drop unspecified columns (e.g. already-binary ones)
        verbose_feature_names_out=False,
    )

    # We fit on binary-encoded df (binary cols passed through as-is via a passthrough)
    bin_cols = [c for c in BINARY_COLS if c in df.columns]
    extra = [("bin", "passthrough", bin_cols)] if bin_cols else []

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("nom", nominal_pipeline, nom_cols),
        ] + extra,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X = df.drop(columns=[TARGET_COL], errors="ignore")
    preprocessor.fit(X)
    logger.info("Preprocessor fitted.")
    return preprocessor


# ── 5. Transform ──────────────────────────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract human-readable feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn
        names: list[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if transformer == "passthrough":
                names.extend(cols)
            elif hasattr(transformer, "get_feature_names_out"):
                names.extend(transformer.get_feature_names_out(cols).tolist())
            else:
                names.extend(cols)
        return names


def preprocess(df: pd.DataFrame, preprocessor: ColumnTransformer) -> tuple[np.ndarray, list[str]]:
    """
    Apply a *fitted* preprocessor to *df* and return (X_array, feature_names).
    """
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    X_arr = preprocessor.transform(X)
    feat_names = get_feature_names(preprocessor)
    return X_arr, feat_names


# ── 6. Full pipeline ───────────────────────────────────────────────────────────

def run_preprocessing_pipeline(
    raw_path: Path = RAW_DATASET_PATH,
    save_artefacts: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str], pd.DataFrame]:
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor, feature_names, cleaned_df
    """
    # Load & clean
    raw_df   = load_raw_data(raw_path)
    clean_df = clean_data(raw_df)
    clean_df = _binary_encode(clean_df)

    # Save processed CSV for reference
    if save_artefacts:
        clean_df.to_csv(PROC_DATASET_PATH, index=False)
        logger.info("Processed dataset saved → %s", PROC_DATASET_PATH)

    # Split before fitting the preprocessor (avoid data leakage)
    X = clean_df.drop(columns=[TARGET_COL])
    y = clean_df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Train: %d rows | Test: %d rows | Churn rate train=%.2f%% test=%.2f%%",
        len(y_train), len(y_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )

    # Fit preprocessor on training data only
    preprocessor = build_preprocessor(X_train_raw.assign(**{TARGET_COL: y_train}))

    X_train = preprocessor.transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)
    feature_names = get_feature_names(preprocessor)

    logger.info("Feature matrix shape: train=%s  test=%s", X_train.shape, X_test.shape)

    # Save artefacts
    if save_artefacts:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor,  MODEL_DIR / PREPROCESSOR_FILENAME)
        joblib.dump(feature_names, MODEL_DIR / FEATURE_NAMES_FILENAME)
        logger.info("Preprocessor & feature names saved → %s", MODEL_DIR)

    return X_train, X_test, y_train, y_test, preprocessor, feature_names, clean_df
