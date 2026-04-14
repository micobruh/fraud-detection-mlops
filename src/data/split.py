from typing import Tuple, List
import pandas as pd
from tqdm import tqdm
import logging
from ..utils import TARGET_COLUMN, TIME_COLUMN, START_DATE, setup_logging

logger = logging.getLogger(__name__)

def temporal_balanced_train_test_split(
    df: pd.DataFrame,
    time_col: str = TIME_COLUMN,
    target_col: str = TARGET_COLUMN,
    train_ratio: float = 0.8,
    train_search_width: float = 0.05,
    min_test_pos: int = 200,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into training and test sets based on temporal order and target balance."""
    # Sort the data by time and reset the index
    df = df.sort_values(time_col)
    n = len(df)
    y = df[target_col].to_numpy()

    # Define the range of possible split points to search for the best balance between training and test sets.
    train_range = range(
        max(1, int(n * (train_ratio - train_search_width))),
        min(n - 2, int(n * (train_ratio + train_search_width))) + 1
    )

    # Iterate over the possible split points and calculate the target rate in the training and test sets.
    # best = None
    train_end = 0
    best_score = float("inf")

    for i in tqdm(train_range):
        y_train = y[: i]
        y_test = y[i: ]

        if len(y_test) == 0:
            continue

        n_test_pos = y_test.sum()
        if n_test_pos < min_test_pos:
            continue

        train_target_rate = y_train.mean()
        test_target_rate = y_test.mean()
        score = abs(train_target_rate - test_target_rate)

        if score < best_score:
            best_score = score
            train_end = i
    
    df_main = df.iloc[: train_end]
    df_local_test = df.iloc[train_end :]
    return df_main, df_local_test


def temporal_train_val_split(df_main: pd.DataFrame, time_col: str = TIME_COLUMN) -> List[Tuple[List[int], List[int]]]:
    """Create expanding temporal train/validation splits based on month."""
    df_temp = df_main.copy()

    # Convert TransactionDT to month number since START_DATE
    start_date = pd.to_datetime(START_DATE)
    df_temp['DT_Month'] = start_date + pd.to_timedelta(df_temp[time_col], unit='s')
    df_temp['DT_Month'] = (
        (df_temp['DT_Month'].dt.year - start_date.year) * 12
        + df_temp['DT_Month'].dt.month
    )

    # Group by month and get the list of TransactionIDs for each month
    month_to_ids = (
        df_temp
        .groupby('DT_Month')
        .apply(lambda x: x.index.tolist())
        .to_dict()
    )
    month_to_ids = dict(sorted(month_to_ids.items()))

    # For each month, use it as the validation set and all previous months as the training set.
    train_val_id_pairs = []
    for val_month in sorted(month_to_ids.keys()):
        val_ids = month_to_ids[val_month]
        train_ids = []
        for m in month_to_ids:
            if m < val_month:
                train_ids.extend(month_to_ids[m])   
        train_val_id_pairs.append((train_ids, val_ids)) 

    return train_val_id_pairs