from __future__ import annotations

import argparse
import ast
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from ..utils import MLFLOW_TEST_EXPERIMENT_NAME, ROOT_DIR, resolve_project_path


logger = logging.getLogger(__name__)

DEFAULT_REGISTERED_MODEL_NAME = "FraudDetectionXGBoostChampion"
DEFAULT_ALIAS = "champion"
DEFAULT_LOGGED_MODEL_NAME = "final_candidate"
DEFAULT_TEST_PERFORMANCE_PATH = "artifacts/model_test_performance.csv"
DEFAULT_METADATA_PATH = "artifacts/champion_model.json"


def configure_tracking_uri(tracking_uri: str | None = None) -> str:
    uri = tracking_uri or mlflow.get_tracking_uri()
    if uri in {"", None, "file:./mlruns", "./mlruns", "mlruns"}:
        uri = (ROOT_DIR / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    return uri


def parse_best_params(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    return ast.literal_eval(value)


def load_final_test_result(
    test_performance_path: str | Path = DEFAULT_TEST_PERFORMANCE_PATH,
) -> dict[str, Any]:
    results_path = resolve_project_path(test_performance_path)
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError(f"No test results found in {results_path}.")

    if "training_data_scope" in df.columns:
        final_rows = df[df["training_data_scope"] == "train_plus_validation"]
        if not final_rows.empty:
            selected = final_rows.sort_values("roc_auc", ascending=False).iloc[0]
        else:
            selected = df.sort_values("roc_auc", ascending=False).iloc[0]
    else:
        selected = df.sort_values("roc_auc", ascending=False).iloc[0]

    result = selected.to_dict()
    result["best_params"] = parse_best_params(result.get("best_params"))
    return result


def get_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    return experiment.experiment_id


def find_latest_test_run(
    client: MlflowClient,
    experiment_id: str,
    run_name: str = "online-streaming-test",
) -> str:
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=20,
    )
    for run in runs:
        if run.info.status == "FINISHED" and "final_test_roc_auc" in run.data.metrics:
            return run.info.run_id
    raise ValueError(f"No finished MLflow run named {run_name!r} with final test metrics was found.")


def find_logged_model_for_run(
    client: MlflowClient,
    experiment_id: str,
    run_id: str,
    logged_model_name: str = DEFAULT_LOGGED_MODEL_NAME,
) -> Any:
    logged_models = client.search_logged_models(
        experiment_ids=[experiment_id],
        max_results=1000,
    )
    candidates = [
        model
        for model in logged_models
        if model.source_run_id == run_id and model.name == logged_model_name
    ]
    if not candidates:
        raise ValueError(
            f"No logged model named {logged_model_name!r} found for MLflow run {run_id}."
        )
    return sorted(candidates, key=lambda model: model.last_updated_timestamp)[-1]


def localize_file_uri(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return uri

    path = Path(unquote(parsed.path))
    if path.exists():
        return path.as_uri()

    parts = path.parts
    if "mlruns" in parts:
        suffix = Path(*parts[parts.index("mlruns") + 1 :])
        local_path = ROOT_DIR / "mlruns" / suffix
        if local_path.exists():
            return local_path.as_uri()

    return uri


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            model_name,
            tags={
                "project": "fraud-detection-mlops",
                "task": "fraud_detection",
            },
        )


def register_model_version(
    client: MlflowClient,
    model_name: str,
    source: str,
    run_id: str,
    logged_model_id: str | None,
    tags: dict[str, str],
) -> ModelVersion:
    ensure_registered_model(client, model_name)
    return client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        model_id=logged_model_id,
        tags=tags,
    )


def write_champion_metadata(
    metadata: dict[str, Any],
    output_path: str | Path = DEFAULT_METADATA_PATH,
) -> Path:
    path = resolve_project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return path


def optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def promote_champion(
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
    experiment_name: str = MLFLOW_TEST_EXPERIMENT_NAME,
    test_performance_path: str | Path = DEFAULT_TEST_PERFORMANCE_PATH,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
    tracking_uri: str | None = None,
) -> dict[str, Any]:
    tracking_uri = configure_tracking_uri(tracking_uri)
    client = MlflowClient()

    final_result = load_final_test_result(test_performance_path)
    experiment_id = get_experiment_id(client, experiment_name)
    run_id = find_latest_test_run(client, experiment_id)
    logged_model = find_logged_model_for_run(client, experiment_id, run_id)
    model_source = localize_file_uri(logged_model.artifact_location)
    classification_threshold = optional_float(final_result.get("classification_threshold"))
    version_tags = {
        "model_name": str(final_result["model_name"]),
        "feature_set_name": str(final_result["feature_set_name"]),
        "training_data_scope": str(final_result["training_data_scope"]),
        "test_mode": str(final_result["test_mode"]),
        "feature_state_policy": str(final_result.get("selected_feature_state_policy", "not_applicable")),
    }
    if classification_threshold is not None:
        version_tags["classification_threshold"] = str(classification_threshold)

    version = register_model_version(
        client=client,
        model_name=registered_model_name,
        source=model_source,
        run_id=run_id,
        logged_model_id=logged_model.model_id,
        tags=version_tags,
    )
    client.set_registered_model_alias(registered_model_name, alias, version.version)

    model_uri = f"models:/{registered_model_name}@{alias}"
    metadata = {
        "registered_model_name": registered_model_name,
        "registered_model_version": int(version.version),
        "alias": alias,
        "model_uri": model_uri,
        "source_model_uri": logged_model.model_uri,
        "source_artifact_uri": model_source,
        "source_run_id": run_id,
        "source_logged_model_id": logged_model.model_id,
        "tracking_uri": tracking_uri,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": final_result["model_name"],
        "feature_set_name": final_result["feature_set_name"],
        "training_data_scope": final_result["training_data_scope"],
        "feature_state_policy": final_result.get("selected_feature_state_policy", "not_applicable"),
        "test_mode": final_result["test_mode"],
        "streaming_batch_size": (
            None
            if pd.isna(final_result.get("streaming_batch_size"))
            else int(final_result["streaming_batch_size"])
        ),
        "classification_threshold": classification_threshold,
        "test_metrics": {
            "roc_auc": float(final_result["roc_auc"]),
            "average_precision": float(final_result["average_precision"]),
            "f1": float(final_result["f1"]),
            "recall": float(final_result["recall"]),
            "precision": float(final_result["precision"]),
            "accuracy": float(final_result["accuracy"]),
        },
        "best_params": final_result["best_params"],
    }
    metadata_file = write_champion_metadata(metadata, metadata_path)
    logger.info("Promoted champion model to %s version %s.", registered_model_name, version.version)
    logger.info("Champion metadata written to %s.", metadata_file)
    return metadata


def load_champion_metadata(
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
) -> dict[str, Any]:
    path = resolve_project_path(metadata_path)
    return json.loads(path.read_text())


def load_model_from_mlflow(
    model_uri: str | None = None,
    metadata_path: str | Path = DEFAULT_METADATA_PATH,
):
    if model_uri is None:
        metadata = load_champion_metadata(metadata_path)
        configure_tracking_uri(metadata.get("tracking_uri"))
        model_uri = metadata["model_uri"]
    return mlflow.sklearn.load_model(model_uri)


def get_latest_model_version(
    registered_model_name: str = DEFAULT_REGISTERED_MODEL_NAME,
    alias: str = DEFAULT_ALIAS,
) -> ModelVersion:
    client = MlflowClient()
    return client.get_model_version_by_alias(registered_model_name, alias)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote the final tested model to the MLflow registry.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote_parser = subparsers.add_parser("promote", help="Register and alias the final tested model.")
    promote_parser.add_argument("--registered-model-name", default=DEFAULT_REGISTERED_MODEL_NAME)
    promote_parser.add_argument("--alias", default=DEFAULT_ALIAS)
    promote_parser.add_argument("--experiment-name", default=MLFLOW_TEST_EXPERIMENT_NAME)
    promote_parser.add_argument("--test-performance-path", default=DEFAULT_TEST_PERFORMANCE_PATH)
    promote_parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    promote_parser.add_argument("--tracking-uri", default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.command == "promote":
        metadata = promote_champion(
            registered_model_name=args.registered_model_name,
            alias=args.alias,
            experiment_name=args.experiment_name,
            test_performance_path=args.test_performance_path,
            metadata_path=args.metadata_path,
            tracking_uri=args.tracking_uri,
        )
        print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
