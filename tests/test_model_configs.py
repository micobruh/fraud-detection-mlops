from src.models.train import get_candidate_configs
from src.pipelines.training_flow import (
    normalize_feature_set_names,
    normalize_search_smote_options,
)
from src.utils import DEFAULT_FEATURE_SET, FEATURE_SETS
from sklearn.base import clone


def test_candidate_configs_build():
    configs = get_candidate_configs(
        selected_columns=["TransactionAmt"],
        feature_set_name=DEFAULT_FEATURE_SET,
        search_smote=False,
    )

    assert configs
    assert all("name" in c for c in configs)
    assert all("pipeline" in c for c in configs)
    assert all("param_distributions" in c for c in configs)


def test_training_comparison_defaults_cover_feature_sets_and_smote_modes():
    assert normalize_feature_set_names(None) == list(FEATURE_SETS)
    assert normalize_search_smote_options(None) == [False, True]


def test_engineered_feature_pipeline_is_sklearn_cloneable():
    configs = get_candidate_configs(
        selected_columns=["TransactionAmt", "card1", "addr1"],
        feature_set_name="base_selected_v_engineered",
        search_smote=False,
    )

    clone(configs[0]["pipeline"])
