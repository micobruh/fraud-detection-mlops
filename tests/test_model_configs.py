from src.models.train import get_candidate_configs
from src.utils import DEFAULT_FEATURE_SET


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
