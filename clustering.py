import io

import joblib
import streamlit as st
from sklearn.metrics import silhouette_score

from data_loader import aggregate_features, load_builtin_dataset, load_uploaded_dataset
from model import attach_labels, cari_k_optimal, preprocess_for_model, run_kmeans


def run_builtin_clustering(n_sample: int) -> None:
    """Jalankan clustering menggunakan dataset bawaan (Indore OLA)."""
    with st.spinner("⏳ Memuat dataset bawaan (Indore OLA)…"):
        raw = load_builtin_dataset()

        total_records = len(raw)
        success_count = (raw["Booking Status"] == "Success").sum()
        cancel_cust   = (raw["Booking Status"] == "Canceled by Customer").sum()
        cancel_drv    = (raw["Booking Status"] == "Canceled by Driver").sum()
        st.session_state.raw_stats = {
            "total": total_records,
            "success": success_count,
            "cancel_cust": cancel_cust,
            "cancel_drv": cancel_drv,
        }

    _process_and_save(raw, n_sample=n_sample, label="Bawaan (Indore OLA)")


def run_uploaded_clustering(uploaded_xlsx, n_sample: int) -> None:
    """Jalankan clustering menggunakan dataset yang di-upload user."""
    with st.spinner("⏳ Memproses dataset kustom yang diunggah…"):
        raw = load_uploaded_dataset(uploaded_xlsx.read(), uploaded_xlsx.name)
        st.session_state.raw_stats = None

    _process_and_save(raw, n_sample=n_sample, label="Kustom")


def run_custom_clustering(uploaded_xlsx, uploaded_pkl) -> None:
    """Jalankan clustering kustom (dengan opsional model PKL yang di-upload)."""
    with st.spinner("⏳ Memproses dataset kustom…"):
        raw_c = load_uploaded_dataset(uploaded_xlsx.read(), uploaded_xlsx.name)
        df_base_c = aggregate_features(raw_c)
        df_s_c, df_enc_c, df_scaled_c, scaler_c = preprocess_for_model(
            df_base_c, n_sample=len(df_base_c)
        )

        if uploaded_pkl is not None:
            km_c = joblib.load(io.BytesIO(uploaded_pkl.read()))
            for col in km_c.feature_names_in_:
                if col not in df_scaled_c.columns:
                    df_scaled_c[col] = 0
            df_scaled_c = df_scaled_c[km_c.feature_names_in_]
            labels_c = km_c.predict(df_scaled_c)
            k_clusters_c = km_c.n_clusters
            best_score_c = silhouette_score(df_scaled_c, labels_c)
        else:
            k_clusters_c, best_score_c = cari_k_optimal(df_scaled_c)
            km_c = run_kmeans(df_scaled_c, k_clusters_c)
            labels_c = km_c.labels_

        df_result_c = attach_labels(df_s_c, labels_c, k_clusters_c)
        st.session_state.result      = df_result_c
        st.session_state.model_km    = km_c
        st.session_state.scaler      = scaler_c
        st.session_state.df_enc_cols = list(df_enc_c.columns)
        st.session_state.k_clusters  = k_clusters_c
        st.session_state.best_score  = best_score_c

    st.success(
        f"Clustering dataset kustom selesai — K = **{k_clusters_c}** "
        f"(Silhouette Score = **{best_score_c:.4f}**)"
    )


# ─── INTERNAL HELPER ────────────────────────────────────────────────────────

def _process_and_save(raw, n_sample: int, label: str) -> None:
    df_base = aggregate_features(raw)
    df_s, df_enc, df_scaled, scaler = preprocess_for_model(df_base, n_sample)

    with st.spinner("⏳ Mencari K optimal dan membuat klaster…"):
        k_clusters, best_score = cari_k_optimal(df_scaled)
        km = run_kmeans(df_scaled, k_clusters)
        labels = km.labels_

    df_result = attach_labels(df_s, labels, k_clusters)
    st.session_state.result         = df_result
    st.session_state.model_km       = km
    st.session_state.scaler         = scaler
    st.session_state.df_enc_cols    = list(df_enc.columns)
    st.session_state.k_clusters     = k_clusters
    st.session_state.best_score     = best_score
    st.session_state.auto_run_done  = True

    st.success(
        f"Clustering Dataset {label} selesai — K optimal = **{k_clusters}** "
        f"(Silhouette Score = **{best_score:.4f}**)"
    )
