"""Evidently AI population drift monitor component for ClinicaGuard."""

import os
import streamlit as st
import pandas as pd
from src.core.data_generator import ClinicalDataGenerator
from src.services.drift_monitor import ClinicalDriftMonitor


def render_drift_dashboard(base_dir: str):
    st.subheader("📡 Evidently AI Clinical Population Drift Monitor")
    st.caption("Covariate Shift Detection on Demographic, Physiological & Lab Biomarkers")

    st.markdown("""
    In clinical production deployments, distribution shift occurs due to seasonal viral waves, changing hospital referral demographics, or evolving clinician documentation styles. Evidently AI monitors whether the inference cohort distribution significantly deviates from training baseline data.
    """)

    scenario = st.radio(
        "Select Test Scenario:",
        ["Baseline In-Distribution Cohort (Stable)", "Shifted Population (Aging Cohort + Renal Surge)"]
    )

    if st.button("🚀 Run Evidently AI Drift Evaluation", type="primary"):
        with st.spinner("Calculating Wasserstein distance and Kolmogorov-Smirnov drift statistics..."):
            gen = ClinicalDataGenerator(seed=42)
            ref_df = gen.generate_patient_batch(count=150)

            if scenario.startswith("Baseline"):
                curr_df = gen.generate_patient_batch(count=150)
            else:
                curr_df = gen.generate_patient_batch(count=150)
                # Shift distribution
                curr_df["age"] = curr_df["age"] + 12
                curr_df["creatinine"] = curr_df["creatinine"] * 1.8
                curr_df["emergency_visits_1yr"] = curr_df["emergency_visits_1yr"] + 3
                curr_df["spo2"] = curr_df["spo2"] - 4

            reports_dir = os.path.join(base_dir, "outputs", "drift_reports")
            monitor = ClinicalDriftMonitor(reports_dir=reports_dir)
            summary = monitor.evaluate_cohort_drift(ref_df, curr_df, report_name="live_drift_eval")

            st.divider()
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)

            drift_status = "⚠️ DRIFT DETECTED" if summary["dataset_drift_detected"] else "✅ POPULATION STABLE"
            dcol1.metric("Overall Cohort Status", drift_status)
            dcol2.metric("Drift Share", f"{summary['drift_share_pct']}%")
            dcol3.metric("Drifted Biomarkers", f"{summary['drifted_features_count']} / {summary['total_features_monitored']}")
            dcol4.metric("Engine", "Evidently AI 0.7.23")

            st.write("#### Feature-Level Covariate Drift Audit")
            feat_rows = []
            for f in summary.get("feature_metrics", []):
                feat_rows.append({
                    "Clinical Biomarker": f["feature"],
                    "Reference Baseline Mean": f["reference_mean"],
                    "Current Cohort Mean": f["current_mean"],
                    "Standardized Z-Shift": f"{f['z_score_shift']:.3f}",
                    "Drift Flag": "🚨 DRIFT" if f["drift_detected"] else "✅ STABLE",
                })
            st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

            if os.path.exists(summary.get("html_report_path", "")):
                st.success(f"Evidently AI HTML Report generated: `{summary['html_report_path']}`")
