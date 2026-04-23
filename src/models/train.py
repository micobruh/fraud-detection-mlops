from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..pipelines import build_feature_pipeline
from ..utils import RANDOM_STATE


def build_full_pipeline(feature_pipeline, model, sampler="passthrough"):
    return ImbPipeline([
        ("features", feature_pipeline),
        ("sampler", sampler),        
        ("model", model),
    ])


def get_candidate_configs(
    selected_columns,
    feature_set_name,
    search_smote: bool = False,
    random_state: int = RANDOM_STATE,
):
    feature_pipeline = build_feature_pipeline(selected_columns, feature_set_name)
    sampler_options = ["passthrough"]
    if search_smote:
        sampler_options.append(SMOTE(random_state=random_state))

    return [
        {
            "name": "lightgbm",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=LGBMClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": sampler_options,
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__num_leaves": [31, 63],
                "model__max_depth": [-1, 10],
            },
        },
        {
            "name": "xgboost",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=XGBClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    eval_metric="logloss",
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": sampler_options,
                "model__n_estimators": [200, 500],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [4, 6, 8],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
            },
        },
        {
            "name": "random_forest",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": sampler_options,
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 10],
            },
        },
    ]


def build_pipeline_from_config(
    selected_columns,
    feature_set_name,
    model_name: str,
    best_params: dict,
    search_smote: bool = False,
    random_state: int = RANDOM_STATE,
):
    candidate_configs = get_candidate_configs(
        selected_columns,
        feature_set_name,
        search_smote=search_smote,
        random_state=random_state,
    )
    matching_config = next(
        (config for config in candidate_configs if config["name"] == model_name),
        None,
    )

    if matching_config is None:
        raise ValueError(f"Unknown model configuration: {model_name}")

    pipeline = matching_config["pipeline"]
    pipeline.set_params(**best_params)
    return pipeline
