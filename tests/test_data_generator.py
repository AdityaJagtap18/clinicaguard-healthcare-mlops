"""Tests for synthetic clinical EHR data generator."""

import pytest
import pandas as pd
from src.core.data_generator import ClinicalDataGenerator


def test_clinical_data_generation_columns_and_counts():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=150)
    assert len(df) == 150
    assert "patient_id" in df.columns
    assert "discharge_summary" in df.columns
    assert "readmitted_30d" in df.columns

    # Verify key vitals and lab panels
    assert "sbp" in df.columns
    assert "creatinine" in df.columns
    assert "spo2" in df.columns
    assert "charlson_score" in df.columns
    assert "emergency_visits_1yr" in df.columns

    # Target distribution
    readmit_rate = df["readmitted_30d"].mean()
    assert 0.10 <= readmit_rate <= 0.40  # Realistic clinical readmission prevalence


def test_clinical_vitals_physiologic_bounds():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=100)
    assert (df["sbp"] >= 80).all() and (df["sbp"] <= 240).all()
    assert (df["hr"] >= 40).all() and (df["hr"] <= 180).all()
    assert (df["spo2"] >= 80).all() and (df["spo2"] <= 100).all()
    assert (df["temperature_c"] >= 35.0).all() and (df["temperature_c"] <= 41.0).all()
