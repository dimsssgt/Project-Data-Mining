import warnings

import streamlit as st

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clustering Pola Konsumsi Transportasi Online Indore OLA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── IMPORTS LOKAL ───────────────────────────────────────────────────────────
from styles import inject_styles
from sidebar import render_sidebar
from clustering import run_builtin_clustering, run_uploaded_clustering, run_custom_clustering
from tabs import ringkasan, visualisasi, evaluasi, data as tab_data

# ─── STYLES ──────────────────────────────────────────────────────────────────
inject_styles()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
sidebar = render_sidebar()
run_btn        = sidebar["run_btn"]
run_custom_btn = sidebar["run_custom_btn"]
uploaded_xlsx  = sidebar["uploaded_xlsx"]
uploaded_pkl   = sidebar["uploaded_pkl"]
n_sample       = sidebar["n_sample"]

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
for key in ["result", "model_km", "scaler", "df_enc_cols", "k_clusters",
            "best_score", "raw_stats", "auto_run_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>Sistem Clustering Pola Konsumsi Pengguna Transportasi Online</h1>
    <p>Dataset: Indore OLA Dataset &nbsp;|&nbsp; Algoritma: K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

# ─── CLUSTERING ──────────────────────────────────────────────────────────────
auto_trigger = st.session_state.auto_run_done is None
should_run   = run_btn or auto_trigger

if should_run:
    if uploaded_xlsx is not None:
        run_uploaded_clustering(uploaded_xlsx, n_sample)
    else:
        run_builtin_clustering(n_sample)

if run_custom_btn and uploaded_xlsx is not None:
    run_custom_clustering(uploaded_xlsx, uploaded_pkl)

# ─── DISPLAY RESULT ──────────────────────────────────────────────────────────
df_result = st.session_state.result

if df_result is not None:
    k_val    = st.session_state.k_clusters
    raw_s    = st.session_state.raw_stats

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan", "📈 Visualisasi", "📄 Evaluasi Model", "📋 Data Hasil"
    ])

    with tab1:
        ringkasan.render(df_result, k_val, raw_s)

    with tab2:
        visualisasi.render(df_result, k_val)

    with tab3:
        evaluasi.render(df_result, k_val)

    with tab4:
        tab_data.render(df_result)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.78rem'>"
    "Sistem Clustering Pola Konsumsi Transportasi Online &nbsp;|&nbsp; "
    "K-Means Clustering &nbsp;|&nbsp; Dataset: Indore OLA 2025"
    "</div>",
    unsafe_allow_html=True,
)
