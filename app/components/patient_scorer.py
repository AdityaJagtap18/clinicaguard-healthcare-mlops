"""Bedside patient admission risk evaluator component for ClinicaGuard."""

import streamlit as st
import pandas as pd
import numpy as np
from src.core.data_generator import ClinicalDataGenerator
from src.core.embedding_engine import ClinicalEmbeddingEngine
from src.core.vector_store import ClinicalVectorStore
from src.core.multimodal_model import ReadmissionPredictionPipeline


@st.cache_resource
def load_scoring_pipeline():
    """Initializes and trains the baseline and multimodal models on synthetic clinical cohort."""
    gen = ClinicalDataGenerator(seed=42)
    df = gen.generate_patient_batch(count=600)
    emb_engine = ClinicalEmbeddingEngine()
    embeddings = emb_engine.encode(df["discharge_summary"].tolist())

    store = ClinicalVectorStore()
    store.insert_patient_records(df, embeddings)
    priors = store.get_cluster_priors_batch(embeddings, k=10)

    pipe = ReadmissionPredictionPipeline(random_state=42)
    X, y = pipe.prepare_tabular_features(df)
    pipe.train_tabular_baseline(X, y, use_cv=False)
    pipe.train_multimodal_model(X, y, priors, use_cv=False)

    return pipe, store, emb_engine, df


def render_patient_scorer():
    st.subheader("🩺 Bedside Multimodal 30-Day Readmission Risk Scorer")
    st.caption("EHR Structured Vitals + pgvector Clinical Note Semantic Similarity Search")

    pipe, store, emb_engine, cohort_df = load_scoring_pipeline()

    st.markdown("#### Patient Admission Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.slider("Patient Age (Safe Harbor Cap: 89):", 21, 89, 72)
        gender = st.selectbox("Gender:", ["M", "F"])
        emergency_visits = st.number_input("ED Visits in Past 12 Months:", 0, 15, 2)
    with col2:
        sbp = st.slider("Systolic BP (mmHg):", 80, 220, 148)
        dbp = st.slider("Diastolic BP (mmHg):", 50, 130, 88)
        hr = st.slider("Heart Rate (bpm):", 45, 160, 92)
    with col3:
        spo2 = st.slider("Oxygen Saturation (SpO2 %):", 80, 100, 91)
        creatinine = st.number_input("Serum Creatinine (mg/dL):", 0.5, 8.0, 2.1, step=0.1)
        bun = st.number_input("Blood Urea Nitrogen (mg/dL):", 5.0, 120.0, 38.0, step=1.0)
    with col4:
        los_days = st.slider("Hospital Length of Stay (Days):", 1, 30, 6)
        icu_hours = st.number_input("ICU Stay (Hours):", 0, 200, 24)
        charlson_score = st.slider("Charlson Comorbidity Score:", 0, 15, 5)

    st.write("#### Clinician Discharge Summary (Unstructured Text)")
    sample_notes = [
        "Elderly patient presented with severe acute decompensated heart failure with bilateral pulmonary edema. EF 30%. History of CKD with elevated creatinine. High risk for post-discharge fluid overload.",
        "Routine observation following mild community-acquired pneumonia. Course uneventful; vitals within normal parameters throughout hospitalization. Tolerating oral diet well. Low suspicion for rapid readmission.",
        "Patient admitted for uncontrolled Type 2 Diabetes with hyperglycemic crisis. Creatinine slightly elevated. Responded to insulin sliding scale. Discharged home on adjusted regimen with diabetes educator consult."
    ]
    selected_sample = st.selectbox("Select Clinical Discharge Preset or Customize Below:", sample_notes)
    discharge_text = st.text_area("Discharge Note Text:", value=selected_sample, height=100)

    if st.button("🔬 Compute Readmission Risk Score", type="primary", use_container_width=True):
        with st.spinner("Embedding note with Sentence-Transformers & querying pgvector HNSW index..."):
            # 1. Embed note
            q_emb = emb_engine.encode(discharge_text)[0]

            # 2. Query pgvector for cohort prior
            cluster_prior, matches = store.compute_cohort_readmission_prior(q_emb, k=10)

            # 3. Assemble features
            patient_features = {
                "age": age, "sbp": sbp, "dbp": dbp, "hr": hr, "rr": 18, "spo2": spo2,
                "temperature_c": 37.1, "glucose": 140.0, "creatinine": creatinine, "bun": bun,
                "hemoglobin": 11.5, "wbc": 9.5, "sodium": 138.0, "potassium": 4.3, "platelets": 210,
                "los_days": los_days, "icu_stay_hours": icu_hours,
                "emergency_visits_1yr": emergency_visits, "charlson_score": charlson_score,
                "has_congestive_heart_failure": 1 if "heart" in discharge_text.lower() else 0,
                "has_chronic_pulmonary": 1 if "copd" in discharge_text.lower() else 0,
                "has_renal_failure": 1 if creatinine > 1.8 else 0,
                "has_diabetes_uncomplicated": 1 if "diabetes" in discharge_text.lower() else 0,
            }

            # 4. Predict
            res = pipe.predict_single_patient(patient_features, cluster_prior=cluster_prior)

            st.divider()
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)

            tier = res["risk_tier"]
            tier_color = "red" if tier == "HIGH" else ("orange" if tier == "MODERATE" else "green")
            rcol1.markdown(f"### Risk Tier: :{tier_color}[{tier}]")

            rcol2.metric(
                "Multimodal Readmission Prob",
                f"{res['multimodal_hybrid_prob'] * 100:.1f}%",
                delta=f"{res['probability_shift'] * 100:+.1f}% vs Tabular"
            )
            rcol3.metric("Tabular-Only Baseline Prob", f"{res['tabular_baseline_prob'] * 100:.1f}%")
            rcol4.metric("pgvector Cohort Prior P(Readmit)", f"{cluster_prior * 100:.1f}%")

            st.markdown("""
            > **Multimodal Diagnostic Insight**: The tabular model evaluates vitals and lab panels in isolation. By vectorizing the clinician discharge summary and performing a $k$-NN search across historical admissions via `pgvector`, ClinicaGuard detects semantic clinical acuity (such as decompensated cardiomyopathy and fluid overload mentions), shifting the predicted risk accordingly.
            """)

            st.write("#### Nearest Historical Clinical Cohort Matches (pgvector $k$-NN)")
            match_data = [
                {
                    "Patient ID": m.patient_id,
                    "Admission ID": m.admission_id,
                    "Cosine Similarity": f"{m.similarity_score:.4f}",
                    "Readmitted (30d)": "⚠️ YES" if m.readmitted_30d == 1 else "✅ NO",
                    "Discharge Summary Excerpt": m.deidentified_note[:120] + "..."
                }
                for m in matches
            ]
            st.dataframe(pd.DataFrame(match_data), use_container_width=True, hide_index=True)
