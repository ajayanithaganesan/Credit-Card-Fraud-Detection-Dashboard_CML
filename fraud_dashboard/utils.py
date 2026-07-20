"""
fraud_dashboard.utils
---------------------
Backend logic for the Credit Card Fraud Detection Dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import boto3
import joblib
import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

MODEL_CANDIDATES: List[str] = [
    "model.pkl",
    "random_forest_fraud_detector.pkl",
]
MODEL_PATH = PROJECT_ROOT / MODEL_CANDIDATES[0]
SCALER_PATH = PROJECT_ROOT / "scaler.pkl"
METRICS_LOG_FILE = PROJECT_ROOT / "logs" / "model_metrics.log"
METRICS_LOGGER_NAME = "fraud_metrics"
SCALED_COLUMNS: List[str] = ["Time", "Amount", "LogAmount"]
SAMPLE_DATA_PATH = PROJECT_ROOT / "sample_data" / "sample_transactions.csv"
SAMPLE_DATA_BUCKET_ENV = "SAMPLE_DATA_BUCKET"
SAMPLE_DATA_KEY_ENV = "SAMPLE_DATA_KEY"
AWS_REGION_ENV = "AWS_REGION"
AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
LOG_AMOUNT_COL = "LogAmount"
FEATURE_COLUMNS: List[str] = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", LOG_AMOUNT_COL]
)
PREDICTION_COL = "Prediction"
CONFIDENCE_COL = "Confidence (%)"
FRAUD_LABEL = "🔴 Fraud"
GENUINE_LABEL = "🟢 Genuine"


def _get_metrics_logger():
    logger = logging.getLogger(METRICS_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    try:
        METRICS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(METRICS_LOG_FILE)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass

    return logger


def _log_inference_metrics(
    model,
    row_count: int,
    feature_prep_seconds: float,
    prediction_seconds: float,
    probability_seconds: float,
    total_latency_seconds: float,
    fraud_predictions: int,
) -> None:
    logger = _get_metrics_logger()
    throughput = row_count / max(prediction_seconds + probability_seconds, 1e-9)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": str(uuid.uuid4()),
        "model": model.__class__.__name__ if model is not None else "unknown",
        "rows": row_count,
        "feature_prep_ms": round(feature_prep_seconds * 1000, 3),
        "prediction_ms": round(prediction_seconds * 1000, 3),
        "probability_ms": round(probability_seconds * 1000, 3),
        "total_latency_ms": round(total_latency_seconds * 1000, 3),
        "throughput": round(throughput, 3),
        "fraud_predictions": fraud_predictions,
    }

    logger.info(json.dumps(payload, sort_keys=True))


def load_model():
    for name in MODEL_CANDIDATES:
        candidate = PROJECT_ROOT / name
        if candidate.exists():
            return joblib.load(candidate)

    raise FileNotFoundError(
        "Trained model not found. Please place your model file in the "
        f"project root, named one of: {', '.join(MODEL_CANDIDATES)}."
    )


def load_scaler():
    if SCALER_PATH.exists():
        return joblib.load(SCALER_PATH)
    return None


def get_sample_data_s3_config() -> tuple[str | None, str | None, str | None]:
    """Return the configured S3 bucket, object key, and region for the sample CSV."""
    bucket = os.getenv(SAMPLE_DATA_BUCKET_ENV, "").strip() or None
    key = os.getenv(SAMPLE_DATA_KEY_ENV, "").strip() or None
    region = (
        os.getenv(AWS_REGION_ENV, "").strip()
        or os.getenv(AWS_DEFAULT_REGION_ENV, "").strip()
        or None
    )
    return bucket, key, region


def sample_data_s3_configured() -> bool:
    """Return True when the sample CSV S3 location has been configured."""
    bucket, key, _ = get_sample_data_s3_config()
    return bool(bucket and key)


def download_sample_csv_bytes_from_s3(
    bucket: str, key: str, region: str | None = None
) -> bytes:
    """Download the sample CSV object from S3 and return the raw bytes."""
    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region

    s3 = boto3.client("s3", **client_kwargs)
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()


def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def prepare_features(df: pd.DataFrame, model=None, scaler=None) -> pd.DataFrame:
    working = df.copy()
    working = working.apply(pd.to_numeric, errors="coerce")

    if LOG_AMOUNT_COL not in working.columns and "Amount" in working.columns:
        working[LOG_AMOUNT_COL] = np.log1p(working["Amount"].clip(lower=0))

    if model is not None and hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        feature_order = FEATURE_COLUMNS

    for col in feature_order:
        if col not in working.columns:
            working[col] = 0.0

    features = working[feature_order].fillna(0.0)

    if scaler is not None:
        scale_cols = list(getattr(scaler, "feature_names_in_", SCALED_COLUMNS))
        scale_cols = [c for c in scale_cols if c in features.columns]
        if scale_cols:
            features[scale_cols] = scaler.transform(features[scale_cols])

    return features


def missing_feature_columns(df: pd.DataFrame) -> List[str]:
    return [
        c for c in FEATURE_COLUMNS if c != LOG_AMOUNT_COL and c not in df.columns
    ]


def run_predictions(model, df: pd.DataFrame, scaler=None) -> pd.DataFrame:
    overall_start = perf_counter()

    feature_start = perf_counter()
    X = prepare_features(df, model=model, scaler=scaler)
    feature_prep_seconds = perf_counter() - feature_start

    prediction_start = perf_counter()
    predictions = model.predict(X)
    prediction_seconds = perf_counter() - prediction_start

    probability_start = perf_counter()
    proba = model.predict_proba(X)
    probability_seconds = perf_counter() - probability_start

    total_latency_seconds = perf_counter() - overall_start
    fraud_probability = proba[:, 1]

    result = df.copy()
    result[PREDICTION_COL] = np.where(
        predictions == 1, FRAUD_LABEL, GENUINE_LABEL
    )
    result[CONFIDENCE_COL] = np.round(fraud_probability * 100, 2)

    try:
        _log_inference_metrics(
            model=model,
            row_count=len(X),
            feature_prep_seconds=feature_prep_seconds,
            prediction_seconds=prediction_seconds,
            probability_seconds=probability_seconds,
            total_latency_seconds=total_latency_seconds,
            fraud_predictions=int(np.sum(predictions == 1)),
        )
    except Exception:  # noqa: BLE001
        pass

    return result


def build_summary(result: pd.DataFrame) -> Dict[str, float]:
    total = len(result)
    fraud = int((result[PREDICTION_COL] == FRAUD_LABEL).sum())
    genuine = total - fraud
    fraud_pct = round((fraud / total) * 100, 2) if total else 0.0
    return {
        "total": total,
        "fraud": fraud,
        "genuine": genuine,
        "fraud_pct": fraud_pct,
    }


def top_high_risk(result: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    ranked = (
        result.reset_index()
        .rename(columns={"index": "Transaction"})
        .sort_values(CONFIDENCE_COL, ascending=False)
        .head(n)
    )
    return ranked[["Transaction", CONFIDENCE_COL, PREDICTION_COL]]


def filter_view(result: pd.DataFrame, view: str) -> pd.DataFrame:
    if view == "Fraud Only":
        return result[result[PREDICTION_COL] == FRAUD_LABEL]
    if view == "Genuine Only":
        return result[result[PREDICTION_COL] == GENUINE_LABEL]
    if view == "High Risk (>90%)":
        return result[result[CONFIDENCE_COL] > 90]
    return result


def to_csv_bytes(result: pd.DataFrame) -> bytes:
    return result.to_csv(index=False).encode("utf-8")
