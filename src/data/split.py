from typing import Tuple, List
import pandas as pd
import logging
from ..features import (
    determine_columns
)
from ..utils import (
    START_DATE,
    DEFAULT_FEATURE_SET,
    TARGET_COLUMN, 
    TIME_COLUMN,   
    DROP_COLUMNS  
)

logger = logging.getLogger(__name__)


def temporal_train_val_test_split(
    df: pd.DataFrame, 
    feature_set_name: str = DEFAULT_FEATURE_SET,
    threshold: float = 0.65,
    v_columns_cache_path: str | None = "artifacts/selected_v_columns.json",
) -> Tuple[
        List[Tuple[List[int], List[int]]], 
        List[str],
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame
    ]:
    """Create expanding temporal train/validation/test splits based on month."""
    df_temp = df[[TIME_COLUMN]].sort_values(TIME_COLUMN)

    # Convert TransactionDT to month number since START_DATE
    start_date = pd.to_datetime(START_DATE)
    df_temp['DT_Month'] = start_date + pd.to_timedelta(df_temp[TIME_COLUMN], unit='s')
    df_temp['DT_Month'] = (
        (df_temp['DT_Month'].dt.year - start_date.year) * 12
        + df_temp['DT_Month'].dt.month
    )

    # Group by month and get the list of TransactionIDs for each month
    month_to_ids = {
        month: df_temp.index.take(pos_idx).tolist()
        for month, pos_idx in sorted(df_temp.groupby("DT_Month").indices.items())
    }

    # For each month, use it as the validation set and all previous months as the training set.
    all_train_val_id_pairs = []
    for val_month in sorted(month_to_ids.keys()):
        val_ids = month_to_ids[val_month]
        train_ids = []
        for m in month_to_ids:
            if m < val_month:
                train_ids.extend(month_to_ids[m])   
        all_train_val_id_pairs.append((train_ids, val_ids)) 

    # There are 6 months in total for the whole data
    # First 4 months are used for initial training data and batch offline cross validation
    # 5th month is used for streaming online validation window
    # 6th month is used for streaming online test window
    columns = determine_columns(df, feature_set_name, threshold, v_columns_cache_path)
    df_main = df.loc[all_train_val_id_pairs[-2][0]]
    df_stream_val = df.loc[all_train_val_id_pairs[-2][1]]
    df_stream_test = df.loc[all_train_val_id_pairs[-1][1]]

    # sklearn CV splitters expect positional indices relative to X_train/y_train,
    # not TransactionID labels from the DataFrame index.
    train_index_to_pos = {idx: pos for pos, idx in enumerate(df_main.index)}
    cv_splits = []
    for train_ids, val_ids in all_train_val_id_pairs[1:-2]:
        train_pos = [train_index_to_pos[idx] for idx in train_ids if idx in train_index_to_pos]
        val_pos = [train_index_to_pos[idx] for idx in val_ids if idx in train_index_to_pos]
        cv_splits.append((train_pos, val_pos))

    available_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
    model_input_columns = list(dict.fromkeys(columns + available_drop_columns))

    X_train = df_main[model_input_columns]
    X_stream_val = df_stream_val[model_input_columns]
    X_stream_test = df_stream_test[model_input_columns]
    y_train = df_main[TARGET_COLUMN]
    y_stream_val = df_stream_val[TARGET_COLUMN]
    y_stream_test = df_stream_test[TARGET_COLUMN]   

    return (
        cv_splits, 
        columns,
        X_train, 
        X_stream_val, 
        X_stream_test, 
        y_train, 
        y_stream_val, 
        y_stream_test
    )
