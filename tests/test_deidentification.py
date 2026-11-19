"""Tests for HIPAA Safe Harbor clinical text de-identification."""

import pytest
import pandas as pd
from src.core.data_generator import ClinicalDataGenerator


def test_clean_data_passes_hipaa_audit():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=100)
    audit = ClinicalDataGenerator.audit_hipaa_safe_harbor(df)

    assert audit["hipaa_compliant"] is True
    assert audit["age_cap_enforced"] is True
    assert len(audit["violations"]) == 0


def test_hipaa_audit_catches_leaked_ssn():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=10)
    # Inject leaked SSN into first note
    df.loc[0, "discharge_summary"] += " Patient SSN: 123-45-6789."

    audit = ClinicalDataGenerator.audit_hipaa_safe_harbor(df)
    assert audit["hipaa_compliant"] is False
    assert any("SSN" in v for v in audit["violations"])


def test_hipaa_audit_catches_uncapped_age():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=10)
    # Inject uncapped age > 89
    df.loc[0, "age"] = 94

    audit = ClinicalDataGenerator.audit_hipaa_safe_harbor(df)
    assert audit["hipaa_compliant"] is False
    assert any("age > 89" in v for v in audit["violations"])
