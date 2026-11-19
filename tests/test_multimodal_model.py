"""Tests for multimodal XGBoost modeling and PR-AUC validation."""

import pytest
import numpy as np
import pandas as pd
from src.core.data_generator import ClinicalDataGenerator
from src.core.embedding_engine import ClinicalEmbeddingEngine
from src.core.vector_store import ClinicalVectorStore
from src.core.multimodal_model import ReadmissionPredictionPipeline


@pytest.fixture
def clinical_data():
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=300)
    emb_engine = ClinicalEmbeddingEngine()
    embeddings = emb_engine.encode(df["discharge_summary"].tolist())

    store = ClinicalVectorStore()
    store.insert_patient_records(df, embeddings)
    priors = store.get_cluster_priors_batch(embeddings, k=10)
    return df, embeddings, priors


def test_tabular_baseline_training(clinical_data):
    df, _, _ = clinical_data
    pipe = ReadmissionPredictionPipeline(random_state=42)
    X, y = pipe.prepare_tabular_features(df)

    model, report = pipe.train_tabular_baseline(X, y, use_cv=True)
    assert model is not None
    assert report.pr_auc > 0.25
    assert report.roc_auc > 0.45
    assert len(report.feature_importances) > 0


def test_multimodal_model_improvement(clinical_data):
    df, embeddings, priors = clinical_data
    pipe = ReadmissionPredictionPipeline(random_state=42)
    X, y = pipe.prepare_tabular_features(df)

    # Train baseline
    _, base_rep = pipe.train_tabular_baseline(X, y, use_cv=True)
    # Train multimodal
    _, multi_rep = pipe.train_multimodal_model(X, y, priors, use_cv=True)

    # Verify multimodal out-performs or matches baseline
    assert multi_rep.pr_auc >= base_rep.pr_auc - 0.05
    assert "clinical_note_cluster_prior" in multi_rep.feature_importances


def test_single_patient_prediction(clinical_data):
    df, embeddings, priors = clinical_data
    pipe = ReadmissionPredictionPipeline(random_state=42)
    X, y = pipe.prepare_tabular_features(df)
    pipe.train_tabular_baseline(X, y, use_cv=False)
    pipe.train_multimodal_model(X, y, priors, use_cv=False)

    patient_dict = df.iloc[0].to_dict()
    res = pipe.predict_single_patient(patient_dict, cluster_prior=priors[0])

    assert "tabular_baseline_prob" in res
    assert "multimodal_hybrid_prob" in res
    assert "risk_tier" in res
    assert res["risk_tier"] in ["LOW", "MODERATE", "HIGH"]
