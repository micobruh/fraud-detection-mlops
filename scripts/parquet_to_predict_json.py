import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import TARGET_COLUMN


def dataframe_to_prediction_payload(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    records_df = df.astype(object).where(pd.notna(df), None)
    return {"records": records_df.to_dict(orient="records")}


def convert_parquet_to_predict_json(
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    include_target: bool = False,
) -> Path:
    input_file = Path(input_path)
    output_file = Path(output_path)

    df = pd.read_parquet(input_file)
    if not include_target and TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be None or >= 1")
        df = df.head(limit)

    payload = dataframe_to_prediction_payload(df)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, allow_nan=False))
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a parquet file into the JSON payload expected by the FastAPI /predict endpoint."
    )
    parser.add_argument("input_path", type=Path, help="Input parquet file.")
    parser.add_argument("output_path", type=Path, help="Output JSON file.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to include in the JSON payload.",
    )
    parser.add_argument(
        "--include-target",
        action="store_true",
        help=f"Keep the {TARGET_COLUMN!r} column if it exists. By default it is dropped for inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_file = convert_parquet_to_predict_json(
        args.input_path,
        args.output_path,
        limit=args.limit,
        include_target=args.include_target,
    )
    print(output_file)


if __name__ == "__main__":
    main()
