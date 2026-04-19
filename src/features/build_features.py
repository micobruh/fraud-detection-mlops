from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from ..utils import TIME_COLUMN
import logging

logger = logging.getLogger(__name__)


class DataFrameOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: list[str],
        handle_unknown: str = "use_encoded_value",
        unknown_value: int = -1,
        encoded_missing_value: int = -2,
        dtype: str = "float32",
    ) -> None:
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DataFrameOrdinalEncoder":
        self._validate_input(X)

        self.encoder_ = OrdinalEncoder(
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value,
            encoded_missing_value=self.encoded_missing_value,
            dtype=self.dtype,
        )
        self.encoder_.fit(X[self.columns])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        self._validate_input(X)

        X = X.copy()
        X[self.columns] = self.encoder_.transform(X[self.columns])
        return X

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else self.columns

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DataFrameOrdinalEncoder expects a pandas DataFrame.")

        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "encoder_"):
            raise ValueError("DataFrameOrdinalEncoder is not fitted yet. Call fit() first.")
        

class NumericShiftFillTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: list[str],
        exclude: list[str] | None = None,
        fill_value: float = -1.0,
        dtype: str = "float32",
    ) -> None:
        self.columns = columns
        self.exclude = exclude if exclude is not None else []
        self.fill_value = fill_value
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: Any = None) -> "NumericShiftFillTransformer":
        self._validate_input(X)

        self.columns_to_transform_ = [
            col for col in self.columns if col not in self.exclude
        ]

        self.min_values_ = {}
        for col in self.columns_to_transform_:
            col_min = X[col].min(skipna=True)
            if pd.isna(col_min):
                col_min = 0.0
            self.min_values_[col] = float(col_min)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        self._validate_input(X)

        X = X.copy()

        for col in self.columns_to_transform_:
            X[col] = X[col] - np.float32(self.min_values_[col])
            X[col] = X[col].fillna(self.fill_value).astype(self.dtype)

        return X

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NumericShiftFillTransformer expects a pandas DataFrame.")

        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "min_values_"):
            raise ValueError("NumericShiftFillTransformer is not fitted yet. Call fit() first.")
        

class DColumnNormalizer(BaseEstimator, TransformerMixin):
    """
    Add normalized D columns to the input DataFrame.

    For each selected D column:
        D{i}_normalized = floor(D{i} - TransactionDT / seconds_in_day) + offset

    The transformer returns the full augmented DataFrame so downstream
    transformers can use the newly created columns.
    """

    def __init__(
        self,
        d_indices: list[int] | None = None,
        exclude: tuple[int, ...] = (9,),
        time_col: str = TIME_COLUMN,
        offset: float = 1000.0,
        seconds_in_day: int = 24 * 60 * 60,
        dtype: str = "float32",
    ) -> None:
        self.d_indices = d_indices if d_indices is not None else list(range(1, 16))
        self.exclude = exclude
        self.time_col = time_col
        self.offset = offset
        self.seconds_in_day = seconds_in_day
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y=None) -> "DColumnNormalizer":
        self._validate_input(X)
        self.created_columns_ = [
            f"D{i}_normalized"
            for i in self.d_indices
            if i not in self.exclude
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        self._validate_input(X)

        X = X.copy()

        time_scaled = X[self.time_col] / np.float32(self.seconds_in_day)

        for i in self.d_indices:
            if i in self.exclude:
                continue

            source_col = f"D{i}"
            new_col = f"{source_col}_normalized"

            X[new_col] = (
                np.floor(X[source_col] - time_scaled) + self.offset
            ).astype(self.dtype)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return list(input_features) + self.created_columns_

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DColumnNormalizer expects a pandas DataFrame as input.")

        required_cols = [self.time_col] + [
            f"D{i}" for i in self.d_indices if i not in self.exclude
        ]
        missing_cols = [col for col in required_cols if col not in X.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "created_columns_"):
            raise ValueError("DColumnNormalizer is not fitted yet. Call fit() first.")
    

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    sklearn-compatible frequency encoder with optional streaming updates.

    Parameters
    ----------
    columns : list[str]
        Columns to frequency encode.
    missing_value : float, default=-1.0
        Encoded value for missing values.
    unknown_value : float, default=0.0
        Encoded value for unseen non-missing categories.
    dtype : str, default="float32"
        Output dtype for encoded columns.
    update_during_inference : bool, default=False
        If True, streamed batches can update live counts during deployment.
    use_live_state_for_transform : bool, default=False
        If True, transform() uses live updated counts instead of frozen serving counts.
        For production stability, keep this False.
    add_new_columns : bool, default=True
        If True, creates new columns like <col>_freq.
        If False, overwrites the original columns.
    drop_original : bool, default=False
        If True and add_new_columns=True, drops original raw columns after encoding.
    """

    def __init__(
        self,
        columns: list[str],
        missing_value: float = -1.0,
        unknown_value: float = 0.0,
        dtype: str = "float32",
        update_during_inference: bool = False,
        use_live_state_for_transform: bool = False,
        add_new_columns: bool = True,
        drop_original: bool = False,
    ) -> None:
        self.columns = columns
        self.missing_value = missing_value
        self.unknown_value = unknown_value
        self.dtype = dtype
        self.update_during_inference = update_during_inference
        self.use_live_state_for_transform = use_live_state_for_transform
        self.add_new_columns = add_new_columns
        self.drop_original = drop_original

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        """
        Learn frequencies from training data only.
        """
        X = self._validate_input(X)

        self.freq_maps_ = {}
        self.category_counts_ = {}
        self.total_non_missing_ = {}
        self.missing_count_ = {}

        for col in self.columns:
            s = X[col]
            non_missing = s[~s.isna()]
            counts = non_missing.value_counts(dropna=True)

            total = int(counts.sum())

            self.category_counts_[col] = {k: int(v) for k, v in counts.to_dict().items()}
            self.total_non_missing_[col] = total
            self.missing_count_[col] = int(s.isna().sum())

            if total > 0:
                self.freq_maps_[col] = {k: float(v) / total for k, v in self.category_counts_[col].items()}
            else:
                self.freq_maps_[col] = {}

        # Frozen snapshot used for stable serving
        self.serving_category_counts_ = {
            col: dict(self.category_counts_[col]) for col in self.columns
        }
        self.serving_total_non_missing_ = {
            col: int(self.total_non_missing_[col]) for col in self.columns
        }
        self.serving_freq_maps_ = {
            col: dict(self.freq_maps_[col]) for col in self.columns
        }

        self.is_fitted_ = True
        logger.info("Fitted FrequencyEncoder on columns: %s", self.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform using either frozen serving state or live state.
        """
        self._check_is_fitted()
        X = self._validate_input(X).copy()

        use_live = self.use_live_state_for_transform

        for col in self.columns:
            freq_map = self.freq_maps_[col] if use_live else self.serving_freq_maps_[col]
            encoded = X[col].map(freq_map)
            encoded = encoded.where(~X[col].isna(), self.missing_value)
            encoded = encoded.fillna(self.unknown_value).astype(self.dtype)

            out_col = f"{col}_freq" if self.add_new_columns else col
            X[out_col] = encoded

        if self.add_new_columns and self.drop_original:
            X = X.drop(columns=self.columns)

        return X

    def partial_fit(self, X: pd.DataFrame, y: Any = None) -> "FrequencyEncoder":
        """
        Update live counts from streamed data.

        This does NOT change the frozen serving snapshot unless
        refresh_serving_state() is called.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        for col in self.columns:
            counts = defaultdict(int, self.category_counts_[col])

            for value in X[col]:
                if pd.isna(value):
                    self.missing_count_[col] += 1
                else:
                    counts[value] += 1
                    self.total_non_missing_[col] += 1

            self.category_counts_[col] = dict(counts)

            total = self.total_non_missing_[col]
            if total > 0:
                self.freq_maps_[col] = {
                    k: float(v) / total for k, v in self.category_counts_[col].items()
                }
            else:
                self.freq_maps_[col] = {}

        logger.info("Updated live frequency counts from streamed batch.")
        return self

    def transform_stream(self, X: pd.DataFrame, update_after_transform: bool = False) -> pd.DataFrame:
        """
        Transform a streamed batch during deployment.

        Behavior:
        - uses current serving/live state based on use_live_state_for_transform
        - optionally updates live counts AFTER transforming the batch

        This avoids leakage within the same batch.
        """
        X = self.transform(X)

        if update_after_transform and self.update_during_inference:
            raw_X = self._validate_input(X if not self.add_new_columns else X[[c for c in X.columns if c in self.columns]])
            # safer to update from original columns if they are still present
            # if originals were dropped, caller should pass raw batch to partial_fit separately
            if set(self.columns).issubset(raw_X.columns):
                self.partial_fit(raw_X)

        return X

    def refresh_serving_state(self) -> "FrequencyEncoder":
        """
        Promote live counts to the serving snapshot.

        Call this only when you intentionally want deployment predictions
        to start using the updated stream-informed frequencies.
        """
        self._check_is_fitted()

        self.serving_category_counts_ = {
            col: dict(self.category_counts_[col]) for col in self.columns
        }
        self.serving_total_non_missing_ = {
            col: int(self.total_non_missing_[col]) for col in self.columns
        }
        self.serving_freq_maps_ = {
            col: dict(self.freq_maps_[col]) for col in self.columns
        }

        logger.info("Refreshed serving frequency state.")
        return self

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """
        sklearn-compatible output feature names.
        """
        input_features = input_features if input_features is not None else self.columns

        if self.add_new_columns:
            out = list(input_features) + [f"{col}_freq" for col in self.columns]
            if self.drop_original:
                out = [c for c in out if c not in self.columns]
            return out

        return list(input_features)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder expects a pandas DataFrame as input.")

        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return X

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("FrequencyEncoder is not fitted yet. Call fit() first.")


class CombineColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(self.columns) < 2:
            raise ValueError("CombineColumnsTransformer needs at least 2 columns.")

        missing_cols = [col for col in self.columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input: {missing_cols}")

        X = X.copy()

        new_col_name = "_".join(self.columns)

        # Vectorized combination (cleaner & faster)
        combined = X[self.columns].astype(str).agg("_".join, axis=1)

        return pd.DataFrame({new_col_name: combined}, index=X.index)


class UIDAggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Create aggregation features per UID from training data only.

    For each combination of:
        uid column x main column x aggregation
    this transformer learns a mapping on the training set and applies it
    to new data.

    Example generated feature names:
        TransactionAmt_card1_mean
        TransactionAmt_card1_std
        ProductCD_card1_nunique

    Notes
    -----
    - UID columns are used only as lookup keys and are NOT included in output.
    - Output contains only the engineered aggregation features.
    - Unseen UID values at transform time are filled with a fallback value.
    - Missing UID values are treated like unseen keys unless you pre-impute them.

    Parameters
    ----------
    main_columns : list[str]
        Columns whose values will be aggregated.
    uid_columns : list[str]
        Grouping columns used as UID keys.
    aggregations : list[str], default=("mean",)
        Supported examples: "mean", "std", "min", "max", "median", "sum", "count", "nunique".
    fill_value : float, default=-1.0
        Fallback used when a UID was unseen during fitting, or aggregation is missing.
    use_na_sentinel : bool, default=False
        If True, values equal to `na_sentinel` in main columns are treated as missing
        before fitting the aggregation mappings.
    na_sentinel : float | int, default=-1
        Sentinel value in main columns to convert to NaN when use_na_sentinel=True.
    dtype : str, default="float32"
        Output dtype for engineered columns.
    """

    def __init__(
        self,
        main_columns: list[str],
        uid_columns: list[str],
        aggregations: Iterable[str] = ("mean",),
        fill_value: float = -1.0,
        use_na_sentinel: bool = False,
        na_sentinel: float | int = -1,
        dtype: str = "float32",
    ) -> None:
        self.main_columns = main_columns
        self.uid_columns = uid_columns
        self.aggregations = list(aggregations)
        self.fill_value = fill_value
        self.use_na_sentinel = use_na_sentinel
        self.na_sentinel = na_sentinel
        self.dtype = dtype

    def fit(self, X: pd.DataFrame, y: Any = None) -> "UIDAggregationTransformer":
        X = self._validate_input(X).copy()

        self._check_supported_aggregations()

        self.feature_names_out_: list[str] = []
        self.mapping_dicts_: dict[tuple[str, str, str], dict[Any, float]] = {}
        self.global_fallbacks_: dict[tuple[str, str, str], float] = {}

        for main_col in self.main_columns:
            working_main = X[main_col].copy()

            if self.use_na_sentinel:
                working_main = working_main.mask(working_main == self.na_sentinel, np.nan)

            for uid_col in self.uid_columns:
                temp = pd.DataFrame({
                    uid_col: X[uid_col],
                    main_col: working_main,
                })

                for agg in self.aggregations:
                    feature_name = f"{main_col}_{uid_col}_{agg}"
                    self.feature_names_out_.append(feature_name)

                    # Build mapping: uid -> aggregated value
                    if agg == "nunique":
                        grouped = temp.groupby(uid_col, dropna=False)[main_col].nunique(dropna=True)
                        global_fallback = float(grouped.median())
                    else:
                        grouped = temp.groupby(uid_col, dropna=False)[main_col].agg(agg)
                        global_fallback = self._compute_global_fallback(temp[main_col], agg)

                    mapping = grouped.to_dict()

                    # Normalize NaN aggregation outputs to fallback later
                    cleaned_mapping = {}
                    for key, value in mapping.items():
                        if pd.isna(value):
                            cleaned_mapping[key] = float(self.fill_value)
                        else:
                            cleaned_mapping[key] = float(value)

                    self.mapping_dicts_[(main_col, uid_col, agg)] = cleaned_mapping
                    self.global_fallbacks_[(main_col, uid_col, agg)] = float(global_fallback)

        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        X = self._validate_input(X)

        out = pd.DataFrame(index=X.index)

        for main_col in self.main_columns:
            for uid_col in self.uid_columns:
                uid_values = X[uid_col]

                for agg in self.aggregations:
                    feature_name = f"{main_col}_{uid_col}_{agg}"
                    mapping = self.mapping_dicts_[(main_col, uid_col, agg)]
                    fallback = self.global_fallbacks_[(main_col, uid_col, agg)]

                    encoded = uid_values.map(mapping)
                    encoded = encoded.fillna(fallback)
                    encoded = encoded.fillna(self.fill_value).astype(self.dtype)

                    out[feature_name] = encoded

        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self._check_is_fitted()
        return np.asarray(self.feature_names_out_, dtype=object)

    def _validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("UIDAggregationTransformer expects a pandas DataFrame as input.")

        required = set(self.main_columns) | set(self.uid_columns)
        missing = [col for col in required if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(self.main_columns) == 0:
            raise ValueError("main_columns must contain at least one column.")
        if len(self.uid_columns) == 0:
            raise ValueError("uid_columns must contain at least one column.")

        return X

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "is_fitted_") or not self.is_fitted_:
            raise ValueError("UIDAggregationTransformer is not fitted yet. Call fit() first.")

    def _check_supported_aggregations(self) -> None:
        supported = {"mean", "std", "min", "max", "median", "sum", "count", "nunique"}
        invalid = [agg for agg in self.aggregations if agg not in supported]
        if invalid:
            raise ValueError(
                f"Unsupported aggregations: {invalid}. "
                f"Supported aggregations are: {sorted(supported)}"
            )

    def _compute_global_fallback(self, s: pd.Series, agg: str) -> float:
        """
        Fallback used for unseen UIDs at transform time.
        """
        if agg == "mean":
            value = s.mean()
        elif agg == "std":
            value = s.std()
        elif agg == "min":
            value = s.min()
        elif agg == "max":
            value = s.max()
        elif agg == "median":
            value = s.median()
        elif agg == "sum":
            value = s.sum()
        elif agg == "count":
            value = s.count()
        else:
            raise ValueError(f"Unsupported aggregation for global fallback: {agg}")

        if pd.isna(value):
            return float(self.fill_value)
        return float(value)
    

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from a pandas DataFrame.

    Parameters
    ----------
    columns : Iterable[str]
        Columns to drop.
    errors : {"raise", "ignore"}, default="ignore"
        - "raise": error if any column is missing
        - "ignore": drop only existing columns
    copy : bool, default=True
        Whether to operate on a copy of X.
    """

    def __init__(
        self,
        columns: Iterable[str],
        errors: str = "ignore",
        copy: bool = True,
    ) -> None:
        self.columns = list(columns)
        self.errors = errors
        self.copy = copy

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DropColumnsTransformer":
        self._validate_input(X)

        self.existing_columns_ = [c for c in self.columns if c in X.columns]
        self.missing_columns_ = [c for c in self.columns if c not in X.columns]

        if self.errors == "raise" and self.missing_columns_:
            raise ValueError(f"Columns not found during fit: {self.missing_columns_}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        self._validate_input(X)

        X_out = X.copy() if self.copy else X

        if self.errors == "raise":
            missing = [c for c in self.columns if c not in X_out.columns]
            if missing:
                raise ValueError(f"Columns not found during transform: {missing}")
            return X_out.drop(columns=self.columns)

        # errors == "ignore"
        cols_to_drop = [c for c in self.columns if c in X_out.columns]
        return X_out.drop(columns=cols_to_drop)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [c for c in input_features if c not in self.columns]

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DropColumnsTransformer expects a pandas DataFrame.")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "existing_columns_"):
            raise ValueError("Transformer is not fitted yet. Call fit() first.")    