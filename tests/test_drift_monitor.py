"""Tests for Evidently AI data drift monitor."""

import pytest
import pandas as pd
from src.core.data_generator import ClinicalDataGenerator
from src.services.drift_monitor import ClinicalDriftMonitor


def test_drift_monitor_clean_cohorts(tmp_path):
    gen = ClinicalDataGenerator(seed=42)
    ref_df = gen.generate_patient_batch(count=100)
    curr_df = gen.generate_patient_batch(count=100)

    monitor = ClinicalDriftMonitor(reports_dir=str(tmp_path))
    summary = monitor.evaluate_cohort_drift(ref_df, curr_df, report_name="test_clean_drift")

    assert "dataset_drift_detected" in summary
    assert summary["total_features_monitored"] > 0
    assert summary["drift_share_pct"] >= 0.0


def test_drift_monitor_detects_severe_population_shift(tmp_path):
    gen = ClinicalDataGenerator(seed=42)
    ref_df = gen.generate_patient_batch(count=100)
    curr_df = gen.generate_patient_batch(count=100)

    # Inject severe population aging & renal failure surge
    curr_df["age"] = curr_df["age"] + 15
    curr_df["creatinine"] = curr_df["creatinine"] * 2.5
    curr_df["emergency_visits_1yr"] = curr_df["emergency_visits_1yr"] + 4

    monitor = ClinicalDriftMonitor(reports_dir=str(tmp_path))
    summary = monitor.evaluate_cohort_drift(ref_df, curr_df, report_name="test_shifted_drift")

    # Shifted features should be flagged
    flagged = [f["feature"] for f in summary["feature_metrics"] if f["drift_detected"]]
    assert "age" in flagged or "creatinine" in flagged
