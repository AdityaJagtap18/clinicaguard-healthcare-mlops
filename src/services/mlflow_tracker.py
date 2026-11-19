"""MLflow experiment tracking and model registry service for ClinicaGuard.

Logs model hyperparameters, cross-validation metrics (PR-AUC, ROC-AUC, Brier score),
feature importances, and registered model artifacts.
"""

import os
from typing import Any, Dict, List, Optional
import mlflow
import mlflow.xgboost
import pandas as pd


class MLflowTracker:
    """Manages MLflow experiment runs and artifact persistence."""

    def __init__(
        self,
        experiment_name: str = "ClinicaGuard-Readmission-Prediction",
        tracking_uri: Optional[str] = None
    ):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment_name = experiment_name
        self.experiment = mlflow.set_experiment(self.experiment_name)

    def log_model_run(
        self,
        run_name: str,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[Any] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """Logs a single model experiment run to MLflow."""
        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment.experiment_id) as run:
            run_id = run.info.run_id

            # Log tags
            if tags:
                mlflow.set_tags(tags)
            mlflow.set_tag("project", "ClinicaGuard")
            mlflow.set_tag("task", "30-day-readmission")

            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model artifact if provided
            if model is not None:
                try:
                    mlflow.xgboost.log_model(model, artifact_path="model")
                except Exception:
                    pass

            # Log feature importances as artifact
            if feature_importances:
                imp_df = pd.DataFrame(list(feature_importances.items()), columns=["feature", "importance"])
                imp_path = f"outputs_{run_id}_features.csv"
                imp_df.to_csv(imp_path, index=False)
                mlflow.log_artifact(imp_path)
                try:
                    os.remove(imp_path)
                except Exception:
                    pass

            return run_id

    def list_recent_runs(self) -> List[Dict[str, Any]]:
        """Retrieves history of model runs for comparison."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )
        if runs.empty:
            return []

        results = []
        for _, row in runs.iterrows():
            results.append({
                "run_id": row["run_id"],
                "run_name": row.get("tags.mlflow.runName", "Unnamed"),
                "status": row["status"],
                "pr_auc": row.get("metrics.pr_auc", None),
                "roc_auc": row.get("metrics.roc_auc", None),
                "f1": row.get("metrics.f1", None),
                "brier_score": row.get("metrics.brier_score", None),
                "start_time": row["start_time"],
            })
        return results
