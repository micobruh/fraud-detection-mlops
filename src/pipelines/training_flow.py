from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from data.preprocess_codex import (
    ID_COLUMN,
    TARGET_COLUMN,
    INTERIM_DATA_DIR,
    load_interim_data,
    merge_transaction_and_identity,
    reduce_memory_footprint,
)
from src.features.build_features import FeatureConfig, build_modeling_frame


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    metrics: dict[str, float]
    feature_columns: list[str]
    train_rows: int
    valid_rows: int


def time_based_split(
    df: pd.DataFrame,
    *,
    time_column: str = "TransactionDT",
    valid_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(time_column).reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - valid_fraction))
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def build_training_pipeline(
    X: pd.DataFrame,
    *,
    random_state: int = 42,
) -> Pipeline:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=80,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def compute_ranking_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    top_k = max(1, int(len(y_true) * 0.05))
    top_idx = y_score.sort_values(ascending=False).index[:top_k]
    top_precision = float(y_true.loc[top_idx].mean())
    base_rate = float(y_true.mean())
    lift_at_5pct = top_precision / base_rate if base_rate else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "validation_base_rate": base_rate,
        "precision_at_5pct": top_precision,
        "lift_at_5pct": float(lift_at_5pct),
    }


def prepare_training_frame(
    *,
    data_dir: Path | None = None,
    feature_config: FeatureConfig | None = None,
) -> pd.DataFrame:
    transaction_df, identity_df = load_interim_data(data_dir=data_dir or INTERIM_DATA_DIR)
    merged = merge_transaction_and_identity(transaction_df, identity_df)
    features = build_modeling_frame(merged, config=feature_config)
    return reduce_memory_footprint(features)


def run_training_flow(
    *,
    data_dir: Path | None = None,
    valid_fraction: float = 0.2,
    random_state: int = 42,
    feature_config: FeatureConfig | None = None,
) -> TrainingArtifacts:
    df = prepare_training_frame(data_dir=data_dir, feature_config=feature_config)
    train_df, valid_df = time_based_split(df, valid_fraction=valid_fraction)

    drop_columns = [column for column in [ID_COLUMN, TARGET_COLUMN] if column in df.columns]
    X_train = train_df.drop(columns=drop_columns)
    y_train = train_df[TARGET_COLUMN]
    X_valid = valid_df.drop(columns=drop_columns)
    y_valid = valid_df[TARGET_COLUMN]

    pipeline = build_training_pipeline(X_train, random_state=random_state)
    pipeline.fit(X_train, y_train)

    valid_scores = pd.Series(pipeline.predict_proba(X_valid)[:, 1], index=valid_df.index)
    metrics = compute_ranking_metrics(y_valid, valid_scores)

    return TrainingArtifacts(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=X_train.columns.tolist(),
        train_rows=len(train_df),
        valid_rows=len(valid_df),
    )
