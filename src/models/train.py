from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from ..utils import RANDOM_STATE


def build_model():
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=64,
        random_state=42,
    )


def build_full_pipeline(feature_pipeline):
    model = build_model()
    return ImbPipeline([
        ("features", feature_pipeline),
        ("SMOTE", SMOTE(random_state=RANDOM_STATE)),        
        ("model", model),
    ])