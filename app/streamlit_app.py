"""Main Streamlit application for PRJ-03: ClinicaGuard.

Multimodal Clinical Hospital Readmission Prevention Engine.
"""

import os
import sys
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.components.patient_scorer import render_patient_scorer
from app.components.model_comparison import render_model_comparison
from app.components.vector_inspector import render_vector_inspector
from app.components.drift_dashboard import render_drift_dashboard
from app.components.mlflow_explorer import render_mlflow_explorer
from src.core.data_generator import ClinicalDataGenerator

st.set_page_config(
    page_title="ClinicaGuard — Clinical Readmission Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom medical header styling
st.markdown("""
<style>
    .med-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #065F46;
        margin-bottom: 0.2rem;
    }
    .med-sub {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="med-header">🏥 ClinicaGuard: Multimodal Readmission Prevention Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="med-sub">EHR Structured Features + PostgreSQL / pgvector Clinical Notes + MLflow & Evidently AI</div>', unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/hospital-3.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Control Center Sections:",
    [
        "🩺 Bedside Readmission Scorer",
        "⚖️ Model Benchmark (PR-AUC 0.61 ➔ 0.74)",
        "🔍 pgvector Clinical Note Search",
        "📡 Evidently AI Drift Monitor",
        "🧪 MLflow Experiment Registry",
        "🛡️ HIPAA Safe Harbor Audit",
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Approved Stack")
st.sidebar.markdown("""
- **scikit-learn**: Feature Processing & CV
- **XGBoost / LightGBM**: Tabular & Multimodal Models
- **PostgreSQL + pgvector**: HNSW Vector Index
- **MLflow**: Experiment Tracking & Metrics
- **Evidently AI**: Population Drift Monitoring
- **Python**: Core Medical Logic & APIs
""")

if page == "🩺 Bedside Readmission Scorer":
    render_patient_scorer()
elif page == "⚖️ Model Benchmark (PR-AUC 0.61 ➔ 0.74)":
    render_model_comparison()
elif page == "🔍 pgvector Clinical Note Search":
    render_vector_inspector()
elif page == "📡 Evidently AI Drift Monitor":
    render_drift_dashboard(PROJECT_ROOT)
elif page == "🧪 MLflow Experiment Registry":
    render_mlflow_explorer()
elif page == "🛡️ HIPAA Safe Harbor Audit":
    st.subheader("🛡️ HIPAA Safe Harbor De-Identification Compliance Audit")
    st.caption("Verification of 18 PHI Identifiers Removal & Age Capping under 45 CFR § 164.514(b)(2)")

    gen = ClinicalDataGenerator(seed=42)
    cohort = gen.generate_patient_batch(count=200)
    audit = ClinicalDataGenerator.audit_hipaa_safe_harbor(cohort)

    acol1, acol2, acol3 = st.columns(3)
    status_str = "✅ FULLY COMPLIANT" if audit["hipaa_compliant"] else "❌ VIOLATIONS FOUND"
    acol1.metric("HIPAA Safe Harbor Status", status_str)
    acol2.metric("Audited Patient Records", audit["total_records_checked"])
    acol3.metric("Age Cap Enforced (<= 89)", "VERIFIED" if audit["age_cap_enforced"] else "FAILED")

    st.divider()
    st.write("#### Safe Harbor Clinical Record Preview (Redacted & De-identified)")
    display_df = cohort[["patient_id", "admission_id", "age", "gender", "primary_diagnosis", "discharge_summary"]].head(10)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
