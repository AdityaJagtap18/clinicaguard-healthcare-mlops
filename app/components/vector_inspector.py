"""pgvector semantic clinical note search component for ClinicaGuard."""

import streamlit as st
import pandas as pd
from app.components.patient_scorer import load_scoring_pipeline


def render_vector_inspector():
    st.subheader("🔍 PostgreSQL + pgvector Semantic Clinical Search")
    st.caption("384-Dimensional Dense Vector Similarity Queries over De-Identified Hospital Discharge Summaries")

    pipe, store, emb_engine, cohort_df = load_scoring_pipeline()

    st.write("#### Natural Language Clinical Query")
    query_text = st.text_input(
        "Enter clinical symptoms, diagnosis, or patient profile:",
        value="Patient with severe decompensated heart failure, pulmonary congestion, and worsening renal function"
    )
    k_val = st.slider("Number of Nearest Neighbors (k):", 3, 20, 10)

    if st.button("🔎 Search Clinical Vector Database", type="primary"):
        with st.spinner("Computing query embedding & executing HNSW vector search..."):
            q_emb = emb_engine.encode(query_text)[0]
            prior, matches = store.compute_cohort_readmission_prior(q_emb, k=k_val)

            st.divider()
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Database Vectors Monitored", store.size())
            mcol2.metric(f"Top-{k_val} Cohort Readmission Prior", f"{prior * 100:.1f}%")
            readmit_count = sum(m.readmitted_30d for m in matches)
            mcol3.metric("Readmitted Matches", f"{readmit_count} / {len(matches)}")

            st.write(f"#### Top {k_val} Semantically Similar Patient Admissions:")
            results = []
            for i, m in enumerate(matches, 1):
                results.append({
                    "Rank": i,
                    "Patient ID": m.patient_id,
                    "Admission ID": m.admission_id,
                    "Cosine Similarity": f"{m.similarity_score:.4f}",
                    "Readmission Outcome": "⚠️ READMITTED (30d)" if m.readmitted_30d == 1 else "✅ NOT READMITTED",
                    "Discharge Summary Text": m.deidentified_note,
                })
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
