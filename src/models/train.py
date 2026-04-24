from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from ..features import build_feature_pipeline
from ..utils import RANDOM_STATE

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def build_full_pipeline(feature_pipeline, model, sampler="passthrough", extra_steps=None):
    feature_steps = list(feature_pipeline.steps)
    extra_steps = extra_steps or []
    return ImbPipeline([
        *feature_steps,
        *extra_steps,
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

    candidate_configs = [
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
        {
            "name": "extra_trees",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=ExtraTreesClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced",
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": sampler_options,
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 10],
                "model__max_features": ["sqrt", 0.5, 1.0],
            },
        },
        {
            "name": "logistic_regression",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                extra_steps=[("scale", StandardScaler())],
                model=LogisticRegression(
                    random_state=random_state,
                    class_weight="balanced",
                    solver="saga",
                    max_iter=1000,
                    n_jobs=1,
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": sampler_options,
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
        },
        {
            "name": "balanced_random_forest",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=BalancedRandomForestClassifier(
                    random_state=random_state,
                    n_jobs=1,
                    sampling_strategy="all",
                    replacement=True,
                    bootstrap=False,
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": ["passthrough"],
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_split": [2, 10],
            },
        },
        {
            "name": "easy_ensemble",
            "pipeline": build_full_pipeline(
                feature_pipeline=feature_pipeline,
                model=EasyEnsembleClassifier(
                    random_state=random_state,
                    n_jobs=1,
                ),
                sampler="passthrough",
            ),
            "param_distributions": {
                "sampler": ["passthrough"],
                "model__n_estimators": [10, 20],
                "model__sampling_strategy": ["auto", 0.5],
            },
        },
    ]

    if CatBoostClassifier is not None:
        candidate_configs.append(
            {
                "name": "catboost",
                "pipeline": build_full_pipeline(
                    feature_pipeline=feature_pipeline,
                    model=CatBoostClassifier(
                        random_state=random_state,
                        thread_count=1,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        auto_class_weights="Balanced",
                        verbose=False,
                        allow_writing_files=False,
                    ),
                    sampler="passthrough",
                ),
                "param_distributions": {
                    "sampler": sampler_options,
                    "model__iterations": [200, 500],
                    "model__learning_rate": [0.03, 0.05, 0.1],
                    "model__depth": [4, 6, 8],
                    "model__l2_leaf_reg": [1, 3, 10],
                },
            }
        )

    return candidate_configs


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
