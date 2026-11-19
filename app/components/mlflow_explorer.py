"""MLflow experiment tracking and run explorer component for ClinicaGuard."""

import os
import streamlit as st
import pandas as pd
from src.services.mlflow_tracker import MLflowTracker


def render_mlflow_explorer():
    st.subheader("🧪 MLflow Experiment Tracking & Model Registry")
    st.caption("Tracking Model Hyperparameters, PR-AUC / ROC-AUC Metrics & Artifact Lineage")

    tracker = MLflowTracker()

    st.write("#### Logged Model Runs")
    runs = tracker.list_recent_runs()

    if runs:
        run_df = pd.DataFrame(runs)
        st.dataframe(run_df, use_container_width=True, hide_index=True)
    else:
        st.info("No MLflow runs recorded in this session. Running an evaluation below will log runs automatically.")

    st.divider()
    st.write("#### Trigger MLflow Benchmark Run")
    if st.button("🚀 Train & Log Models to MLflow", type="primary"):
        with st.spinner("Training Tabular Baseline & Multimodal models, logging to MLflow..."):
            from app.components.patient_scorer import load_scoring_pipeline
            pipe, store, emb_engine, df = load_scoring_pipeline()
            X, y = pipe.prepare_tabular_features(df)
            embeddings = emb_engine.encode(df["discharge_summary"].tolist())
            priors = store.get_cluster_priors_batch(embeddings, k=10)

            # 1. Baseline Run
            _, base_rep = pipe.train_tabular_baseline(X, y, use_cv=True)
            run1_id = tracker.log_model_run(
                run_name="ClinicaGuard-Tabular-Baseline",
                params={"model_type": "xgboost", "features_count": 45, "use_embeddings": False},
                metrics={"pr_auc": base_rep.pr_auc, "roc_auc": base_rep.roc_auc, "f1": base_rep.f1, "brier_score": base_rep.brier_score},
                feature_importances=base_rep.feature_importances
            )

            # 2. Multimodal Run
            _, multi_rep = pipe.train_multimodal_model(X, y, priors, use_cv=True)
            run2_id = tracker.log_model_run(
                run_name="ClinicaGuard-Multimodal-Hybrid",
                params={"model_type": "xgboost", "features_count": 46, "use_embeddings": True, "knn_neighbors": 10},
                metrics={"pr_auc": multi_rep.pr_auc, "roc_auc": multi_rep.roc_auc, "f1": multi_rep.f1, "brier_score": multi_rep.brier_score},
                feature_importances=multi_rep.feature_importances
            )

            st.success("Both runs successfully tracked in MLflow!")
            r1, r2 = st.columns(2)
            r1.metric("Tabular Run ID", run1_id[:12] + "...", f"PR-AUC: {base_rep.pr_auc:.3f}")
            r2.metric("Multimodal Run ID", run2_id[:12] + "...", f"PR-AUC: {multi_rep.pr_auc:.3f}")
            st.rerun()
