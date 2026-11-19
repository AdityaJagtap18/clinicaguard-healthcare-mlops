"""Tests for MLflow experiment tracking."""

import pytest
import os
from src.services.mlflow_tracker import MLflowTracker


def test_mlflow_run_logging(tmp_path):
    db_file = tmp_path / "test_mlruns.db"
    tracker = MLflowTracker(
        experiment_name="Test-ClinicaGuard-Exp",
        tracking_uri=f"sqlite:///{db_file}"
    )

    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}
    metrics = {"pr_auc": 0.742, "roc_auc": 0.831, "f1": 0.654}
    features = {"creatinine": 0.28, "charlson_score": 0.22, "spo2": 0.15}

    run_id = tracker.log_model_run(
        run_name="test_multimodal_run",
        params=params,
        metrics=metrics,
        feature_importances=features
    )
    assert run_id is not None
    assert len(run_id) > 10

    runs = tracker.list_recent_runs()
    assert len(runs) >= 1
    assert runs[0]["run_id"] == run_id
