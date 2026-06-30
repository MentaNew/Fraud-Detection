import logging

import mlflow


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_mlflow(experiment: str = "fraud_detection") -> mlflow:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment)
    return mlflow
