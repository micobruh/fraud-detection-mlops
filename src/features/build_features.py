from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.preprocess_codex import TARGET_COLUMN, add_missing_indicators


DEFAULT_LOW_CARDINALITY_CATEGORICALS = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_30",
    "id_31",
    "id_33",
]


@dataclass
class FeatureConfig:
    rare_category_min_frequency: float = 0.01
    missing_indicator_threshold: float = 0.2


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "TransactionDT" in result.columns:
        result["TransactionHour"] = ((result["TransactionDT"] // 3600) % 24).astype("int8")
        result["TransactionDay"] = (result["TransactionDT"] // 86400).astype("int16")
        result["TransactionWeek"] = (result["TransactionDT"] // (86400 * 7)).astype("int16")
    return result


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "TransactionAmt" in result.columns:
        result["LogTransactionAmt"] = np.log1p(result["TransactionAmt"])
        result["TransactionAmtCents"] = ((result["TransactionAmt"] * 100) % 100).fillna(-1).astype("int16")
    return result


def add_presence_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    identity_cols = [col for col in ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"] if col in result.columns]
    if identity_cols:
        result["HasIdentityInfo"] = result[identity_cols].notna().any(axis=1).astype("int8")
    return result


def collapse_rare_categories(
    df: pd.DataFrame,
    categorical_columns: list[str],
    *,
    min_frequency: float,
) -> pd.DataFrame:
    result = df.copy()
    for column in categorical_columns:
        if column not in result.columns:
            continue
        counts = result[column].value_counts(normalize=True, dropna=True)
        keep = set(counts[counts >= min_frequency].index)
        result[column] = result[column].where(result[column].isin(keep), other="Other")
    return result


def build_modeling_frame(
    df: pd.DataFrame,
    *,
    config: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Apply lightweight fraud-oriented feature engineering."""
    feature_config = config or FeatureConfig()
    result = df.copy()

    result = add_time_features(result)
    result = add_amount_features(result)
    result = add_presence_features(result)
    result = add_missing_indicators(
        result,
        columns=[col for col in result.columns if col != TARGET_COLUMN],
        threshold=feature_config.missing_indicator_threshold,
    )
    result = collapse_rare_categories(
        result,
        categorical_columns=DEFAULT_LOW_CARDINALITY_CATEGORICALS,
        min_frequency=feature_config.rare_category_min_frequency,
    )
    return result
