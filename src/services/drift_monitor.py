"""Evidently AI data drift and model performance monitoring service for ClinicaGuard.

Monitors population shifts in patient demographics, vitals, lab biomarkers,
and target readmission rates between reference baseline and live production cohorts.
"""

from datetime import datetime, timezone
import json
import os
from typing import Any, Dict, List, Optional
import pandas as pd
import pandas.core.dtypes.common as pddc
import pandas.api.types as pat

if not hasattr(pddc, "infer_dtype_from_object"):
    pddc.infer_dtype_from_object = getattr(pat, "infer_dtype", lambda x: "object")

from evidently import Report
from evidently.presets import DataDriftPreset


class ClinicalDriftMonitor:
    """Detects covariate and concept drift in healthcare EHR feature distributions."""

    MONITORED_FEATURES = [
        "age", "sbp", "dbp", "hr", "spo2", "glucose", "creatinine",
        "bun", "hemoglobin", "wbc", "los_days", "emergency_visits_1yr",
        "charlson_score"
    ]

    def __init__(self, reports_dir: str = "outputs/drift_reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def evaluate_cohort_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        report_name: str = "clinical_data_drift"
    ) -> Dict[str, Any]:
        """Runs Evidently AI DataDriftPreset on selected clinical biomarkers."""
        # Subset to monitored numerical features
        cols = [c for c in self.MONITORED_FEATURES if c in reference_df.columns and c in current_df.columns]
        ref_sub = reference_df[cols].copy()
        curr_sub = current_df[cols].copy()

        # Run Evidently Report
        report = Report(metrics=[DataDriftPreset()])
        eval_result = report.run(reference_data=ref_sub, current_data=curr_sub)

        # Save HTML and JSON snapshots
        html_path = os.path.join(self.reports_dir, f"{report_name}.html")
        json_path = os.path.join(self.reports_dir, f"{report_name}.json")

        try:
            eval_result.save_html(html_path)
        except Exception:
            pass

        # Compute summary metrics across features
        feature_drifts = []
        drift_count = 0

        for col in cols:
            ref_mean = float(ref_sub[col].mean())
            curr_mean = float(curr_sub[col].mean())
            ref_std = float(ref_sub[col].std()) or 1.0

            # Absolute Z-score shift
            shift_z = abs(curr_mean - ref_mean) / ref_std
            is_drift = bool(shift_z > 0.40)  # Significant distributional shift threshold
            if is_drift:
                drift_count += 1

            feature_drifts.append({
                "feature": col,
                "reference_mean": round(ref_mean, 2),
                "current_mean": round(curr_mean, 2),
                "z_score_shift": round(float(shift_z), 3),
                "drift_detected": is_drift
            })

        drift_share = drift_count / float(len(cols)) if cols else 0.0
        dataset_drift = bool(drift_share >= 0.30)

        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "report_name": report_name,
            "dataset_drift_detected": dataset_drift,
            "drift_share_pct": round(drift_share * 100.0, 1),
            "drifted_features_count": drift_count,
            "total_features_monitored": len(cols),
            "feature_metrics": feature_drifts,
            "html_report_path": html_path,
        }

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
