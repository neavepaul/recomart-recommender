"""Local MLflow tracking helpers for RecoMart model runs."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.core import ROOT

DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
DEFAULT_EXPERIMENT = "recomart-recommender"


def _metric_name(name: str) -> str:
    return name.replace("@", "_at_").replace("[", "_").replace("]", "")


def _mlflow():
    try:
        import mlflow
    except ImportError as error:
        raise RuntimeError(
            "MLflow tracking requires the mlflow package; install requirements.txt"
        ) from error
    return mlflow


@contextmanager
def model_run(
    run_name: str,
    parameters: dict[str, Any],
    tags: dict[str, Any] | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment: str = DEFAULT_EXPERIMENT,
) -> Iterator[Any]:
    """Create a local MLflow run and log its parameters and identifying tags."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    clean_tags = {key: str(value) for key, value in (tags or {}).items()}
    with mlflow.start_run(run_name=run_name, tags=clean_tags):
        mlflow.log_params({key: str(value) for key, value in parameters.items()})
        yield mlflow


def numeric_metrics(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten numeric leaves from a nested report into MLflow metrics."""
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[_metric_name(name)] = float(value)
        elif isinstance(value, dict):
            metrics.update(numeric_metrics(value, name))
    return metrics


def log_result(mlflow: Any, result: dict[str, Any], artifact_name: str) -> None:
    metrics = numeric_metrics(result)
    if metrics:
        mlflow.log_metrics(metrics)
    mlflow.log_dict(result, artifact_name)


def log_existing_artifact(mlflow: Any, path: Path, artifact_path: str) -> None:
    if path.exists():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
