import numpy as np

from ..utils import (
    TIME_COLUMN
)


class StreamingPipelineScorer:
    def __init__(self, fitted_pipeline):
        self.pipeline = fitted_pipeline

    @staticmethod
    def _is_inference_passthrough(step):
        return step is None or step == "passthrough" or hasattr(step, "fit_resample")

    def _transform_for_inference(self, step, Xt):
        if self._is_inference_passthrough(step):
            return Xt

        stream_transform = getattr(step, "transform_stream", None)
        if callable(stream_transform):
            return stream_transform(Xt)

        return step.transform(Xt)

    def predict_proba(self, X_batch):
        Xt = X_batch

        for _, step in self.pipeline.steps[: -1]:
            Xt = self._transform_for_inference(step, Xt)

        model = self.pipeline.steps[-1][1]
        return model.predict_proba(Xt)

    def update(self, X_batch):
        Xt = X_batch

        for _, step in self.pipeline.steps[:-1]:
            if self._is_inference_passthrough(step):
                continue

            partial_fit_stream = getattr(step, "partial_fit_stream", None)
            if callable(partial_fit_stream):
                partial_fit_stream(Xt)

            Xt = self._transform_for_inference(step, Xt)


def streaming_predict_scores(fitted_pipeline, X_stream, batch_size=1, stream_update=True):
    scorer = StreamingPipelineScorer(fitted_pipeline)
    X_stream_sorted = X_stream.sort_values(TIME_COLUMN)
    y_scores = []

    for start in range(0, len(X_stream_sorted), batch_size):
        X_batch = X_stream_sorted.iloc[start: start + batch_size]
        y_score = scorer.predict_proba(X_batch)[:, 1]         
        y_scores.extend(y_score)
        if stream_update:
            scorer.update(X_batch)

    y_scores = np.array(y_scores)
    y_preds = np.where(y_scores >= 0.5, 1, 0)
    return y_scores, y_preds


def offline_predict_scores(fitted_pipeline, X):
    y_scores = fitted_pipeline.predict_proba(X)[:, 1]
    y_preds = np.where(y_scores >= 0.5, 1, 0)
    return y_scores, y_preds


def sort_y_labels(X_stream, y_stream):
    X_stream_sorted = X_stream.sort_values(TIME_COLUMN)
    y_stream_sorted = y_stream.loc[X_stream_sorted.index]
    return X_stream_sorted, y_stream_sorted
