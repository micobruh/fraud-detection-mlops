from pathlib import Path

import pandas as pd

from src.utils import START_DATE, TARGET_COLUMN, TIME_COLUMN


SMOKE_FIXTURE_PATH = Path("tests/fixtures/train_smoke.parquet")


def test_training_smoke_fixture_covers_six_months_and_both_classes():
    assert SMOKE_FIXTURE_PATH.exists()

    df = pd.read_parquet(SMOKE_FIXTURE_PATH)
    event_time = pd.to_datetime(START_DATE) + pd.to_timedelta(
        df[TIME_COLUMN],
        unit="s",
    )
    event_months = event_time.dt.to_period("M")

    assert event_months.nunique() == 6
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}

    rows_by_month_and_class = df.groupby([event_months, TARGET_COLUMN]).size()
    assert (rows_by_month_and_class > 0).all()
    assert len(rows_by_month_and_class) == 12
