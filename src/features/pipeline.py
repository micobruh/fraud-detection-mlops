from typing import List

from sklearn.pipeline import Pipeline

from .build_features import (
    DataFrameOrdinalEncoder,
    DColumnNormalizer,
    DropColumnsTransformer,
    NumericShiftFillTransformer,
    UIDAggregationAppendTransformer,
)
from ..utils import (
    CATEGORICAL_COLUMNS,
    DEFAULT_FEATURE_SET,
    DROP_COLUMNS,
    FEATURE_SETS,
    NUMERICAL_COLUMNS,
    UID_AGGREGATION_FUNCTIONS,
    UID_AGGREGATION_MAIN_COLUMNS,
    UID_AGGREGATION_UID_COLUMNS,
)


def build_feature_pipeline(
    selected_columns: List[str],
    feature_set_name: str = DEFAULT_FEATURE_SET,
) -> Pipeline:
    feature_config = FEATURE_SETS.get(feature_set_name)
    if feature_config is None:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    numeric_columns = [col for col in NUMERICAL_COLUMNS if col in selected_columns + DROP_COLUMNS]
    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in selected_columns and col not in DROP_COLUMNS
    ]
    steps = [
        # Keep the feature pipeline row-local and stateless at inference time.
        ("numerical_shift_fill", NumericShiftFillTransformer(numeric_columns)),
        ("normalize_D_columns", DColumnNormalizer()),
    ]

    if feature_config["use_uid_features"]:
        steps.append((
            "append_uid_aggregates",
            UIDAggregationAppendTransformer(
                main_columns=UID_AGGREGATION_MAIN_COLUMNS,
                uid_columns=UID_AGGREGATION_UID_COLUMNS,
                aggregations=UID_AGGREGATION_FUNCTIONS,
                fill_value=-1.0,
                dtype="float32",
            ),
        ))

    steps.extend([
        ("drop_columns", DropColumnsTransformer(DROP_COLUMNS, copy=False)),
        (
            "ordinal_encode",
            DataFrameOrdinalEncoder(
                categorical_columns,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    return Pipeline(steps, verbose=True)
