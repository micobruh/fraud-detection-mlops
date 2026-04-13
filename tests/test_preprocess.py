import pandas as pd

from data.preprocess_codex import (
    add_missing_indicators,
    merge_transaction_and_identity,
    normalize_identity_columns,
    reduce_memory_footprint,
)
from features.build_features_codex import FeatureConfig, build_modeling_frame


def test_normalize_identity_columns_replaces_hyphens():
    df = pd.DataFrame({"TransactionID": [1], "id-01": [10], "id-30": ["Windows"]})

    normalized = normalize_identity_columns(df)

    assert "id_01" in normalized.columns
    assert "id_30" in normalized.columns
    assert "id-01" not in normalized.columns


def test_merge_transaction_and_identity_uses_normalized_names():
    transactions = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1]})
    identity = pd.DataFrame({"TransactionID": [2], "id-31": ["chrome"]})

    merged = merge_transaction_and_identity(transactions, identity)

    assert "id_31" in merged.columns
    assert merged.loc[1, "id_31"] == "chrome"


def test_add_missing_indicators_only_for_high_missing_columns():
    df = pd.DataFrame(
        {
            "mostly_missing": [1.0, None, None, None],
            "mostly_present": [1.0, 2.0, 3.0, None],
        }
    )

    result = add_missing_indicators(df, ["mostly_missing", "mostly_present"], threshold=0.5)

    assert "mostly_missing_missing" in result.columns
    assert "mostly_present_missing" not in result.columns


def test_build_modeling_frame_adds_core_eda_features():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "isFraud": [0, 1, 0],
            "TransactionDT": [3600, 90000, 200000],
            "TransactionAmt": [10.0, 25.5, 99.9],
            "ProductCD": ["W", "C", "rare"],
            "DeviceType": [None, "mobile", "desktop"],
            "id_31": [None, "chrome", None],
        }
    )

    result = build_modeling_frame(df, config=FeatureConfig(rare_category_min_frequency=0.34))

    assert {"TransactionHour", "TransactionDay", "LogTransactionAmt", "HasIdentityInfo"} <= set(result.columns)
    assert "DeviceType_missing" in result.columns
    assert "Other" in set(result["ProductCD"].astype(str))


def test_reduce_memory_footprint_downcasts_and_categorizes():
    df = pd.DataFrame(
        {
            "int_col": pd.Series([1, 2, 3], dtype="int64"),
            "float_col": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
            "obj_col": ["a", "b", "c"],
        }
    )

    result = reduce_memory_footprint(df)

    assert str(result["int_col"].dtype).startswith("int")
    assert result["float_col"].dtype != "float64"
    assert str(result["obj_col"].dtype) == "category"
