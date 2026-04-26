from pathlib import Path
import sys

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import RANDOM_STATE, START_DATE, TARGET_COLUMN, TIME_COLUMN


INPUT_PATH = ROOT_DIR / "data/interim/ieee-fraud-detection/train.parquet"
OUTPUT_PATH = ROOT_DIR / "tests/fixtures/train_smoke.parquet"
MONTHS = 6
ROWS_PER_MONTH_AND_CLASS = 20


def build_smoke_subset(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
    months: int = MONTHS,
    rows_per_month_and_class: int = ROWS_PER_MONTH_AND_CLASS,
) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    event_time = pd.to_datetime(START_DATE) + pd.to_timedelta(df[TIME_COLUMN], unit="s")
    event_month = event_time.dt.to_period("M")

    selected_months = sorted(event_month.unique())[:months]
    if len(selected_months) != months:
        raise ValueError(f"Expected {months} months, found {len(selected_months)}")

    sampled_parts = []
    for month in selected_months:
        month_mask = event_month == month
        for target_value in (0, 1):
            class_rows = df[month_mask & (df[TARGET_COLUMN] == target_value)]
            if class_rows.empty:
                raise ValueError(f"No class {target_value} rows found for month {month}")

            sample_size = min(len(class_rows), rows_per_month_and_class)
            sampled_parts.append(
                class_rows.sample(sample_size, random_state=RANDOM_STATE)
            )

    smoke_df = pd.concat(sampled_parts).sort_values(TIME_COLUMN).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_df.to_parquet(output_path, index=False)
    return smoke_df


def main() -> None:
    smoke_df = build_smoke_subset()
    event_time = pd.to_datetime(START_DATE) + pd.to_timedelta(
        smoke_df[TIME_COLUMN],
        unit="s",
    )
    months = sorted(str(month) for month in event_time.dt.to_period("M").unique())
    class_counts = smoke_df[TARGET_COLUMN].value_counts().sort_index().to_dict()

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Rows: {len(smoke_df)}")
    print(f"Months: {months}")
    print(f"Class counts: {class_counts}")


if __name__ == "__main__":
    main()
