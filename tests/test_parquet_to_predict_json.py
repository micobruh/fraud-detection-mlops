import json

import pandas as pd
import pytest

from scripts.parquet_to_predict_json import convert_parquet_to_predict_json


def test_convert_parquet_to_predict_json_writes_predict_payload(tmp_path):
    input_path = tmp_path / "transactions.parquet"
    output_path = tmp_path / "predict_payload.json"
    df = pd.DataFrame(
        {
            "isFraud": [0, 1],
            "TransactionID": [101, 102],
            "TransactionDT": [1, 2],
            "TransactionAmt": [25.0, None],
            "ProductCD": ["W", None],
        }
    )
    df.to_parquet(input_path, index=False)

    result_path = convert_parquet_to_predict_json(input_path, output_path)

    assert result_path == output_path
    assert json.loads(output_path.read_text()) == {
        "records": [
            {
                "TransactionID": 101,
                "TransactionDT": 1,
                "TransactionAmt": 25.0,
                "ProductCD": "W",
            },
            {
                "TransactionID": 102,
                "TransactionDT": 2,
                "TransactionAmt": None,
                "ProductCD": None,
            },
        ]
    }


def test_convert_parquet_to_predict_json_can_keep_target_for_debugging(tmp_path):
    input_path = tmp_path / "transactions.parquet"
    output_path = tmp_path / "predict_payload.json"
    pd.DataFrame({"isFraud": [1], "TransactionID": [101]}).to_parquet(input_path, index=False)

    convert_parquet_to_predict_json(input_path, output_path, include_target=True)

    assert json.loads(output_path.read_text()) == {
        "records": [{"isFraud": 1, "TransactionID": 101}]
    }


def test_convert_parquet_to_predict_json_can_limit_rows(tmp_path):
    input_path = tmp_path / "transactions.parquet"
    output_path = tmp_path / "predict_payload.json"
    pd.DataFrame({"TransactionID": [101, 102]}).to_parquet(input_path, index=False)

    convert_parquet_to_predict_json(input_path, output_path, limit=1)

    assert json.loads(output_path.read_text()) == {
        "records": [{"TransactionID": 101}]
    }


def test_convert_parquet_to_predict_json_rejects_invalid_limit(tmp_path):
    input_path = tmp_path / "transactions.parquet"
    pd.DataFrame({"TransactionID": [101]}).to_parquet(input_path, index=False)

    with pytest.raises(ValueError, match="limit"):
        convert_parquet_to_predict_json(input_path, tmp_path / "payload.json", limit=0)
