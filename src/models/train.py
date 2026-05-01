from typing import Any, Literal, Sequence, TypedDict, TypeAlias
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler

from ..features import build_feature_pipeline
from ..utils import RANDOM_STATE


ModelName: TypeAlias = Literal[
    "lightgbm",
    "xgboost",
    "random_forest",
    "logistic_regression",
    "balanced_random_forest",
    "easy_ensemble",
    "catboost",
]
ParamDistributions: TypeAlias = dict[str, list[Any]]


class CandidateConfig(TypedDict):
    name: ModelName
    pipeline: ImbPipeline
    param_distributions: ParamDistributions


MODEL_NAMES: tuple[ModelName, ...] = (
    "lightgbm",
    "xgboost",
    "random_forest",
    "logistic_regression",
    "balanced_random_forest",
    "easy_ensemble",
    "catboost",
)


def build_full_pipeline(
    feature_pipeline: SklearnPipeline,
    model: BaseEstimator,
    sampler: Any = "passthrough",
    extra_steps: Sequence[tuple[str, Any]] | None = None,
) -> ImbPipeline:
    feature_steps = list(feature_pipeline.steps)
    extra_steps = extra_steps or []
    return ImbPipeline([
        *feature_steps,
        *extra_steps,
        ("sampler", sampler),        
        ("model", model),
    ])


def build_base_pipeline_for_model(
    selected_columns: Sequence[str],
    feature_set_name: str,
    model_name: ModelName,
    sampler: Any = "passthrough",
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    feature_pipeline = build_feature_pipeline(selected_columns, feature_set_name)

    if model_name == "lightgbm":
        return build_full_pipeline(
            feature_pipeline,
            LGBMClassifier(random_state=random_state, n_jobs=1, class_weight="balanced"),
            sampler=sampler,
        )

    elif model_name == "xgboost":
        return build_full_pipeline(
            feature_pipeline,
            XGBClassifier(random_state=random_state, n_jobs=1, eval_metric="logloss"),
            sampler=sampler,
        )
    
    elif model_name == "random_forest":
        return build_full_pipeline(
            feature_pipeline,
            RandomForestClassifier(random_state=random_state, n_jobs=1, class_weight="balanced_subsample"),
            sampler=sampler,
        )    

    elif model_name == "logistic_regression":
        return build_full_pipeline(
            feature_pipeline,
            LogisticRegression(random_state=random_state, n_jobs=1, class_weight="balanced", solver="saga", max_iter=1000),
            sampler=sampler,
            extra_steps=[("scale", StandardScaler())],
        )  
    
    elif model_name == "balanced_random_forest":
        return build_full_pipeline(
            feature_pipeline,
            BalancedRandomForestClassifier(random_state=random_state, n_jobs=1, sampling_strategy="all", replacement=True, bootstrap=False),
            sampler=sampler,
        )   

    elif model_name == "easy_ensemble":
        return build_full_pipeline(
            feature_pipeline,
            EasyEnsembleClassifier(random_state=random_state, n_jobs=1),
            sampler=sampler,
        )         

    elif model_name == "catboost":
        return build_full_pipeline(
            feature_pipeline,
            CatBoostClassifier(random_state=random_state, thread_count=1, loss_function="Logloss", eval_metric="AUC", auto_class_weights="Balanced", verbose=False, allow_writing_files=False),
            sampler=sampler,
        )   

    else:
        raise ValueError(f"Unknown model name: {model_name}")   


def get_model_param_distributions(
    model_name: ModelName,
    sampler_options: list[Any],
) -> ParamDistributions:
    if model_name == "lightgbm":
        return {
            "sampler": sampler_options,
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 10],
        }

    if model_name == "xgboost":
        return {
            "sampler": sampler_options,
            "model__n_estimators": [200, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }

    if model_name == "random_forest":
        return {
            "sampler": sampler_options,
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 10],
        }

    if model_name == "logistic_regression":
        return {
            "sampler": sampler_options,
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l1", "l2"],
        }

    if model_name == "balanced_random_forest":
        return {
            "sampler": ["passthrough"],
            "model__n_estimators": [200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 10],
        }

    if model_name == "easy_ensemble":
        return {
            "sampler": ["passthrough"],
            "model__n_estimators": [10, 20],
            "model__sampling_strategy": ["auto", 0.5],
        }

    if model_name == "catboost":
        return {
            "sampler": sampler_options,
            "model__iterations": [200, 500],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__depth": [4, 6, 8],
            "model__l2_leaf_reg": [1, 3, 10],
        }

    raise ValueError(f"Unknown model name: {model_name}")


def get_candidate_configs(
    selected_columns: Sequence[str],
    feature_set_name: str,
    search_smote: bool = True,
    random_state: int = RANDOM_STATE,
) -> list[CandidateConfig]:
    sampler_options: list[Any] = (
        [SMOTE(random_state=random_state)]
        if search_smote
        else ["passthrough"]
    )

    candidate_configs: list[CandidateConfig] = []

    for model_name in MODEL_NAMES:
        param_distributions = get_model_param_distributions(
            model_name,
            sampler_options=sampler_options,
        )

        if search_smote and all(
            sampler == "passthrough"
            for sampler in param_distributions.get("sampler", [])
        ):
            continue

        candidate_configs.append({
            "name": model_name,
            "pipeline": build_base_pipeline_for_model(
                selected_columns,
                feature_set_name,
                model_name,
                sampler="passthrough",
                random_state=random_state,
            ),
            "param_distributions": param_distributions,
        })

    return candidate_configs


def deserialize_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    params = best_params.copy()
    sampler = params.get("sampler", "passthrough")

    if isinstance(sampler, dict) and sampler.get("type") == "SMOTE":
        params["sampler"] = SMOTE(**sampler.get("params", {}))

    return params


def build_pipeline_from_config(
    selected_columns: Sequence[str],
    feature_set_name: str,
    model_name: ModelName,
    best_params: dict[str, Any],
    random_state: int = RANDOM_STATE,
) -> ImbPipeline:
    best_params = deserialize_best_params(best_params)
    sampler = best_params.get("sampler", "passthrough")

    pipeline = build_base_pipeline_for_model(
        selected_columns,
        feature_set_name,
        model_name,
        sampler=sampler,
        random_state=random_state,
    )

    model_params = {
        key: value
        for key, value in best_params.items()
        if key != "sampler"
    }
    pipeline.set_params(**model_params)

    return pipeline
