import pandas as pd

from src.pipelines.training_flow import (
    compute_ranking_metrics,
    run_training_flow,
    time_based_split,
)


def _write_sample_dataset(tmp_path):
    transaction_df = pd.DataFrame(
        {
            "TransactionID": list(range(1, 11)),
            "isFraud": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
            "TransactionDT": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            "TransactionAmt": [10, 20, 30, 250, 22, 28, 300, 18, 275, 325],
            "ProductCD": ["W", "W", "C", "C", "W", "W", "C", "W", "C", "C"],
            "card4": ["visa", "visa", "mastercard", "mastercard", "visa", "visa", "mastercard", "visa", "mastercard", "mastercard"],
            "card6": ["debit", "debit", "credit", "credit", "debit", "debit", "credit", "debit", "credit", "credit"],
            "P_emaildomain": ["gmail.com", "gmail.com", "outlook.com", "outlook.com", None, None, "gmail.com", None, "outlook.com", "gmail.com"],
            "R_emaildomain": [None, None, "gmail.com", "gmail.com", None, None, "outlook.com", None, "gmail.com", "outlook.com"],
            "M4": [None, None, "M2", "M2", "M0", "M0", "M2", None, "M2", "M2"],
        }
    )
    identity_df = pd.DataFrame(
        {
            "TransactionID": [3, 4, 7, 9, 10],
            "DeviceType": ["desktop", "mobile", "mobile", "mobile", "desktop"],
            "DeviceInfo": ["Windows", "iOS Device", "Android", "Android", "MacOS"],
            "id-30": ["Windows 10", "iOS 11.3.0", "Android 7.0", "Android 7.0", "Mac OS X 10_12_6"],
            "id-31": ["chrome 62.0", "mobile safari generic", "chrome generic", "chrome generic", "safari generic"],
            "id-33": ["1920x1080", "2208x1242", "1334x750", "2208x1242", "2560x1600"],
        }
    )
    transaction_df.to_parquet(tmp_path / "train_transaction.parquet", index=False)
    identity_df.to_parquet(tmp_path / "train_identity.parquet", index=False)


def test_time_based_split_preserves_order():
    df = pd.DataFrame({"TransactionDT": [30, 10, 20], "isFraud": [0, 1, 0]})

    train_df, valid_df = time_based_split(df, valid_fraction=1 / 3)

    assert train_df["TransactionDT"].tolist() == [10, 20]
    assert valid_df["TransactionDT"].tolist() == [30]


def test_compute_ranking_metrics_returns_expected_keys():
    y_true = pd.Series([0, 1, 0, 1, 0, 1])
    y_score = pd.Series([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])

    metrics = compute_ranking_metrics(y_true, y_score)

    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "validation_base_rate",
        "precision_at_5pct",
        "lift_at_5pct",
    }
    assert metrics["roc_auc"] >= 0.5


def test_run_training_flow_end_to_end_on_sample_data(tmp_path):
    _write_sample_dataset(tmp_path)

    artifacts = run_training_flow(data_dir=tmp_path, valid_fraction=0.3, random_state=7)

    assert artifacts.train_rows == 7
    assert artifacts.valid_rows == 3
    assert "TransactionHour" in artifacts.feature_columns
    assert "HasIdentityInfo" in artifacts.feature_columns
    assert 0.0 <= artifacts.metrics["roc_auc"] <= 1.0
    assert 0.0 <= artifacts.metrics["average_precision"] <= 1.0
