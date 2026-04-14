import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from ..utils import TARGET_COLUMN

def remove_high_corr(
    df_main: pd.DataFrame,
    V_group: List[str],
    target_col: str = TARGET_COLUMN,
    threshold: float = 0.65,
) -> List[str]:
    if len(V_group) <= 1:
        return V_group.copy()

    # Absolute correlation matrix
    corr_matrix = df_main[V_group].corr().abs()
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

    # Keep one representative per group: highest nunique
    def score(c):
        # Combine correlation with target and variance to select representative column
        # Higher correlation with target and higher variance is preferred
        target_corr = abs(df_main[c].corr(df_main[target_col]))
        var = df_main[c].var()
        return target_corr + 0.01 * var    

    keep_cols = [max(group, key=lambda c: score(c)) for group in groups]
    return keep_cols

def extract_relevant_V_columns(
    df_main: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    threshold: float = 0.65
) -> List[str]:
    nan_counts = df_main.isna().sum()

    V_groups_with_same_nan_counts = [
        [col for col in cols if col.startswith("V")]
        for null_count, cols in nan_counts.groupby(nan_counts).groups.items()
        if null_count > 0 and len(cols) > 1 and any(col.startswith("V") for col in cols)
    ]

    less_correlated_V_columns = []
    for V_group in tqdm(V_groups_with_same_nan_counts):
        less_correlated_V_columns += remove_high_corr(df_main, V_group, target_col, threshold)
        
    return less_correlated_V_columns    