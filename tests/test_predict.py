import numpy as np
import pandas as pd

from src.models.predict import streaming_predict_scores


class AddOneTransformer:
    def transform(self, X):
        Xt = X.copy()
        Xt["amount"] = Xt["amount"] + 1
        return Xt


class DummyModel:
    def predict_proba(self, X):
        probabilities = np.clip(X["amount"].to_numpy(dtype=float) / 10, 0, 1)
        return np.column_stack([1 - probabilities, probabilities])


class DummyPipeline:
    steps = [
        ("add_one", AddOneTransformer()),
        ("sampler", "passthrough"),
        ("model", DummyModel()),
    ]


def test_streaming_predict_scores_skips_passthrough_sampler():
    X_stream = pd.DataFrame(
        {
            "TransactionDT": [2, 1],
            "amount": [3.0, 1.0],
        }
    )

    y_scores, y_preds = streaming_predict_scores(
        DummyPipeline(),
        X_stream,
        batch_size=1,
        stream_update=True,
    )

    np.testing.assert_allclose(y_scores, [0.2, 0.4])
    np.testing.assert_array_equal(y_preds, [0, 0])
