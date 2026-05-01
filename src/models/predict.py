import numpy as np

from ..utils import (
    TIME_COLUMN
)


class StreamingPipelineScorer:
    def __init__(self, fitted_pipeline):
        self.pipeline = fitted_pipeline

    def predict_proba(self, X_batch):
        Xt = X_batch

        for _, step in self.pipeline.steps[: -1]:
            stream_transform = getattr(step, "transform_stream", None)

            if callable(stream_transform):
                Xt = stream_transform(Xt)
            else:
                Xt = step.transform(Xt)

        model = self.pipeline.steps[-1][1]
        return model.predict_proba(Xt)

    def update(self, X_batch):
        Xt = X_batch

        for _, step in self.pipeline.steps[:-1]:
            partial_fit_stream = getattr(step, "partial_fit_stream", None)
            if callable(partial_fit_stream):
                partial_fit_stream(Xt)

            stream_transform = getattr(step, "transform_stream", None)
            if callable(stream_transform):
                Xt = stream_transform(Xt)
            else:
                Xt = step.transform(Xt)


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
