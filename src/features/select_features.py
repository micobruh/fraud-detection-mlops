import pandas as pd
import numpy as np
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from ..utils import (
    TARGET_COLUMN, 
    BASE_COLUMNS, 
    V_COLUMNS, 
    FEATURE_SETS, 
    DEFAULT_FEATURE_SET
)


logger = logging.getLogger(__name__)


def remove_high_corr(
    df_main: pd.DataFrame,
    V_group: List[str],
    threshold: float = 0.65,
) -> List[str]:
    if len(V_group) <= 1:
        return V_group.copy()

    v_group_df = df_main[V_group]

    # Absolute correlation matrix
    corr_matrix = v_group_df.corr().abs()
    corr_matrix = corr_matrix.fillna(0.0)
    np.fill_diagonal(corr_matrix.values, 1.0)

    # Convert to distance matrix: small distance = high correlation
    dist_matrix = 1 - corr_matrix

    # Condensed distance format required by scipy linkage
    condensed_dist = squareform(dist_matrix.values, checks=False)

    # Hierarchical clustering
    Z = linkage(condensed_dist, method="average")

    # Cut the dendrogram
    cluster_labels = fcluster(Z, t=1 - threshold, criterion="distance")

    # Collect groups
    cluster_map = {}
    for col, label in zip(V_group, cluster_labels):
        cluster_map.setdefault(label, []).append(col)

    groups = list(cluster_map.values())

    # Sort columns inside each group numerically if possible
    def sort_key(x: str):
        digits = "".join(ch for ch in x if ch.isdigit())
        return (x.rstrip(digits), int(digits)) if digits else (x, 0)

    groups = [sorted(group, key=sort_key) for group in groups]
    groups = sorted(groups, key=lambda g: sort_key(g[0]))

    # Precompute scoring inputs once per group instead of recomputing
    # correlation/variance for every candidate column.
    target_corrs = v_group_df.corrwith(df_main[TARGET_COLUMN]).abs().fillna(0.0)
    variances = v_group_df.var().fillna(0.0)

    # Keep one representative per group: highest correlation with target,
    # with variance as a light tie-breaker.
    def score(c):
        return target_corrs[c] + 0.01 * variances[c]

    return [max(group, key=lambda c: score(c)) for group in groups]


def _load_cached_v_columns(
    cache_path: Path,
    threshold: float,
    available_v_columns: List[str],
) -> List[str] | None:
    if not cache_path.exists():
        return None

    try:
        cache_data = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if cache_data.get("threshold") != threshold:
        return None

    if sorted(cache_data.get("available_v_columns", [])) != sorted(available_v_columns):
        return None

    cached_columns = [
        col for col in cache_data.get("columns", [])
        if col in available_v_columns
    ]
    return cached_columns or None


def _write_cached_v_columns(
    cache_path: Path,
    threshold: float,
    available_v_columns: List[str],
    selected_columns: List[str],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "threshold": threshold,
        "available_v_columns": available_v_columns,
        "columns": selected_columns,
    }, indent=2))


def extract_relevant_V_columns(
    df_main: pd.DataFrame,
    threshold: float = 0.65,
    cache_path: str | Path | None = None,
) -> List[str]:
    v_columns_in_df = [col for col in V_COLUMNS if col in df_main.columns]
    if not v_columns_in_df:
        return []

    if cache_path is not None:
        cached_columns = _load_cached_v_columns(Path(cache_path), threshold, v_columns_in_df)
        if cached_columns is not None:
            logger.info("Loaded cached selected V columns from %s", cache_path)
            return cached_columns

    v_df = df_main[v_columns_in_df]
    nan_counts = v_df.isna().sum()

    V_groups_with_same_nan_counts = [
        list(cols)
        for null_count, cols in nan_counts.groupby(nan_counts).groups.items()
        if null_count > 0 and len(cols) > 1
    ]

    less_correlated_V_columns = []
    for V_group in tqdm(V_groups_with_same_nan_counts):
        less_correlated_V_columns.extend(remove_high_corr(df_main, V_group, threshold))

    if cache_path is not None and less_correlated_V_columns:
        _write_cached_v_columns(
            Path(cache_path),
            threshold,
            v_columns_in_df,
            less_correlated_V_columns,
        )
        logger.info("Cached selected V columns to %s", cache_path)

    return less_correlated_V_columns


def determine_columns(
    df_main: pd.DataFrame, 
    feature_set_name: str = DEFAULT_FEATURE_SET,
    threshold: float = 0.65,
    cache_path: str | Path | None = "artifacts/selected_v_columns.json",
) -> List[str]:
    feature_config = FEATURE_SETS.get(feature_set_name)
    if feature_config is None:
        raise ValueError(f"Unknown feature set: {feature_set_name}")

    base_columns = BASE_COLUMNS
    if feature_config["use_selected_v"]:
        v_columns = extract_relevant_V_columns(df_main, threshold, cache_path)
    else:
        v_columns = []
    return base_columns + v_columns
