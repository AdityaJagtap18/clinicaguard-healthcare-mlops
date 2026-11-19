"""Tests for pgvector similarity search and cohort readmission priors."""

import pytest
import numpy as np
import pandas as pd
from src.core.vector_store import ClinicalVectorStore
from src.core.embedding_engine import ClinicalEmbeddingEngine


def test_embedding_engine_dimension():
    engine = ClinicalEmbeddingEngine()
    emb = engine.encode("Patient admitted with acute heart failure and shortness of breath.")
    assert emb.shape == (1, 384)
    # Check L2 normalization
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_vector_store_insert_and_knn():
    engine = ClinicalEmbeddingEngine()
    store = ClinicalVectorStore()

    df = pd.DataFrame({
        "patient_id": [f"PAT-{i:03d}" for i in range(10)],
        "admission_id": [f"ADM-{i:03d}" for i in range(10)],
        "discharge_summary": [f"Discharge note {i} with clinical instructions" for i in range(10)],
        "readmitted_30d": [1 if i % 2 == 0 else 0 for i in range(10)]
    })
    embeddings = engine.encode(df["discharge_summary"].tolist())
    store.insert_patient_records(df, embeddings)

    assert store.size() == 10

    # Query nearest neighbors
    q_vec = embeddings[0]
    matches = store.find_nearest_patients(q_vec, k=5)
    assert len(matches) == 5
    assert matches[0].patient_id == "PAT-000"  # Exact self match highest similarity
    assert matches[0].similarity_score >= 0.99


def test_cohort_readmission_prior_calculation():
    engine = ClinicalEmbeddingEngine()
    store = ClinicalVectorStore()

    # 4 readmitted patients, 1 not readmitted
    df = pd.DataFrame({
        "patient_id": [f"PAT-{i:03d}" for i in range(5)],
        "admission_id": [f"ADM-{i:03d}" for i in range(5)],
        "discharge_summary": [f"Patient note {i}" for i in range(5)],
        "readmitted_30d": [1, 1, 1, 1, 0]
    })
    embeddings = engine.encode(df["discharge_summary"].tolist())
    store.insert_patient_records(df, embeddings)

    prior, matches = store.compute_cohort_readmission_prior(embeddings[0], k=5)
    assert prior == 0.80  # 4/5 = 0.80
    assert len(matches) == 5
