import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import joblib
import io
import os
import warnings

warnings.filterwarnings("ignore")

#  PAGE CONFIG 
st.set_page_config(
    page_title="Clustering Pola Konsumsi Transportasi Online – Indore OLA",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  STYLE 
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 1.9rem; margin: 0; }
    .main-header p  { font-size: 0.9rem; margin: 0.4rem 0 0; opacity: 0.8; }

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .metric-card .label { font-size: 0.78rem; color: #6b7280; font-weight: 500; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e3a8a; }

    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1e3a8a;
        border-left: 4px solid #3b82f6;
        padding-left: 0.6rem; margin: 1.5rem 0 0.8rem;
    }
    .badge-default {
        background: #eff6ff; color: #1d4ed8;
        padding: 0.2rem 0.7rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

#  CONSTANTS 
DATASET_PATH = os.path.join(os.path.dirname(__file__), "indore_ola_dataset_readable.xlsx")

# Pemetaan lengkap semua 50 lokasi ke kategori aktivitas
TUJUAN_MAP = {
    # Kuliah / Pendidikan
    "Bhawarkua": "Kuliah / Pendidikan",
    "Bhanwarkuan": "Kuliah / Pendidikan",
    "Rajendra Nagar": "Kuliah / Pendidikan",
    "Rau": "Kuliah / Pendidikan",
    "Navlakha": "Kuliah / Pendidikan",
    "Geeta Bhawan": "Kuliah / Pendidikan",
    "Silicon City": "Kuliah / Pendidikan",
    "Super Corridor": "Kuliah / Pendidikan",
    "Scheme No. 54": "Kuliah / Pendidikan",
    # Kerja / Meeting
    "MG Road": "Kerja / Meeting",
    "RNT Marg": "Kerja / Meeting",
    "Palasia": "Kerja / Meeting",
    "Vijay Nagar": "Kerja / Meeting",
    "MR 10": "Kerja / Meeting",
    "MR 1": "Kerja / Meeting",
    "MR 2": "Kerja / Meeting",
    "MR 3": "Kerja / Meeting",
    "MR 4": "Kerja / Meeting",
    "MR 9": "Kerja / Meeting",
    "Scheme No. 73": "Kerja / Meeting",
    "Scheme No. 74": "Kerja / Meeting",
    "Scheme No. 71": "Kerja / Meeting",
    "Scheme No. 72": "Kerja / Meeting",
    "Scheme No. 94": "Kerja / Meeting",
    "Scheme No. 114": "Kerja / Meeting",
    "AB Road": "Kerja / Meeting",
    "Aerodrome Road": "Kerja / Meeting",
    "South Tukoganj": "Kerja / Meeting",
    # Belanja / Hiburan
    "Scheme No. 140": "Belanja / Hiburan",
    "Pipliyahana": "Belanja / Hiburan",
    "Tilak Nagar": "Belanja / Hiburan",
    "Malharganj": "Belanja / Hiburan",
    "Annapurna": "Belanja / Hiburan",
    "Sapna Sangeeta": "Belanja / Hiburan",
    "Khajrana": "Belanja / Hiburan",
    "Scheme No. 78": "Belanja / Hiburan",
    # Perumahan / Residensial
    "Nanda Nagar": "Perumahan / Residensial",
    "Saket Nagar": "Perumahan / Residensial",
    "LIG Colony": "Perumahan / Residensial",
    "Lokmanya Nagar": "Perumahan / Residensial",
    "Sudama Nagar": "Perumahan / Residensial",
    "Patel Nagar": "Perumahan / Residensial",
    "Hira Nagar": "Perumahan / Residensial",
    "Ring Road": "Perumahan / Residensial",
    "Lasudia Mori": "Perumahan / Residensial",
    "Bada Gwaltoli": "Perumahan / Residensial",
    "Chhoti Gwaltoli": "Perumahan / Residensial",
    "Banganga": "Perumahan / Residensial",
    "Juni Indore": "Perumahan / Residensial",
    "Mhow Naka": "Perumahan / Residensial",
}

PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]

def tentukan_aktivitas(lokasi):
    return TUJUAN_MAP.get(lokasi.strip(), "Lainnya")

#  DATA LOAD
@st.cache_data(show_spinner=False)
def load_builtin_dataset():
    """Load dataset bawaan dari path lokal."""
    raw = pd.read_excel(DATASET_PATH)
    kolom_raksasa = raw.columns[0]
    raw = raw[kolom_raksasa].astype(str).str.split(",", expand=True)
    raw.columns = kolom_raksasa.split(",")
    raw.columns = raw.columns.str.strip()
    raw["Booking Status"] = raw["Booking Status"].astype(str).str.strip()
    raw["Drop Location"]  = raw["Drop Location"].astype(str).str.strip()
    raw["Tujuan_Aktivitas"] = raw["Drop Location"].apply(tentukan_aktivitas)
    return raw

@st.cache_data(show_spinner=False)
def load_uploaded_dataset(file_bytes, file_name):
    """Load dataset dari file yang di-upload."""
    if file_name.endswith(".csv"):
        raw = pd.read_csv(io.BytesIO(file_bytes))
    else:
        raw = pd.read_excel(io.BytesIO(file_bytes))
    kolom_raksasa = raw.columns[0]
    raw = raw[kolom_raksasa].astype(str).str.split(",", expand=True)
    raw.columns = kolom_raksasa.split(",")
    raw.columns = raw.columns.str.strip()

    required_cols = ["Customer Id", "Booking Status", "Drop Location"]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        st.error(f"Kolom wajib tidak ditemukan: {missing}")
        st.stop()

    raw["Booking Status"] = raw["Booking Status"].astype(str).str.strip()
    raw["Drop Location"]  = raw["Drop Location"].astype(str).str.strip()
    raw["Tujuan_Aktivitas"] = raw["Drop Location"].apply(tentukan_aktivitas)
    return raw

def aggregate_features(raw):
    """Buat fitur per pelanggan dari raw data."""
    success = raw[raw["Booking Status"] == "Success"].copy()
    frek = success.groupby("Customer Id").size().reset_index(name="Frekuensi_Perjalanan")
    tujuan = (
        success.groupby("Customer Id")["Tujuan_Aktivitas"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"Tujuan_Aktivitas": "Tujuan"})
    )
    df = pd.merge(frek, tujuan, on="Customer Id")
    return df

def preprocess_for_model(df, n_sample=1000):
    # Stratified sampling — mempertahankan proporsi tujuan dari dataset asli
    n_target = min(n_sample, len(df))
    df_clean = df.dropna(subset=["Tujuan"]).copy()
    proporsi = df_clean["Tujuan"].value_counts(normalize=True)
    frames = []
    for tujuan_val, frac in proporsi.items():
        grup = df_clean[df_clean["Tujuan"] == tujuan_val]
        n_ambil = max(1, round(n_target * frac))
        frames.append(grup.sample(n=min(n_ambil, len(grup)), random_state=42))
    df_s = pd.concat(frames).head(n_target).copy()
    df_s.set_index("Customer Id", inplace=True)
    Q1, Q3 = df_s["Frekuensi_Perjalanan"].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_s = df_s[
        (df_s["Frekuensi_Perjalanan"] >= Q1 - 1.5 * IQR) &
        (df_s["Frekuensi_Perjalanan"] <= Q3 + 1.5 * IQR)
    ]
    df_enc = pd.get_dummies(df_s, columns=["Tujuan"]).astype(int)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_enc),
        columns=df_enc.columns, index=df_enc.index
    )
    return df_s, df_enc, df_scaled, scaler

def cari_k_optimal(df_scaled, k_min=2, k_max=10):
    best_k, best_score = 2, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score

def run_kmeans(df_scaled, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_scaled)
    return km

def attach_labels(df_s, labels, k):
    df_out = df_s.copy()
    df_out["Klaster"] = [f"Klaster {l+1}" for l in labels]
    return df_out

#  SESSION STATE INIT 
for key in ["result", "model_km", "scaler", "df_enc_cols", "k_clusters",
            "best_score", "raw_stats", "auto_run_done"]:
    if key not in st.session_state:
        st.session_state[key] = None

#  SIDEBAR 
with st.sidebar:
    st.markdown("## Konfigurasi")
    st.markdown("---")

    n_sample = 1000

    # Fitur Tambahan: Upload Dataset Kustom
    st.markdown(" ")
    with st.expander("📁 Upload Dataset Kustom (Opsional)", expanded=False):
        st.caption("Gunakan fitur ini jika ingin mengganti dataset dengan file lain.")
        uploaded_xlsx = st.file_uploader(
            "Upload file dataset (xlsx / csv)",
            type=["xlsx", "csv"],
            key="custom_upload"
        )
        uploaded_pkl = st.file_uploader(
            "Upload model PKL (opsional)",
            type=["pkl"],
            key="custom_pkl"
        )
        use_custom = st.checkbox("Aktifkan dataset kustom", value=False)
        if use_custom and uploaded_xlsx:
            run_custom_btn = st.button("▶ Jalankan dengan Dataset Kustom",
                                       use_container_width=True)
        else:
            run_custom_btn = False
            if use_custom:
                st.info("Upload file dataset terlebih dahulu.")

    run_btn = st.button("Jalankan Clustering", use_container_width=True, type="primary")


#  HEADER 
st.markdown("""
<div class="main-header">
    <h1>Sistem Clustering Pola Konsumsi Pengguna Transportasi Online</h1>
    <p>Dataset: Indore OLA Dataset &nbsp;|&nbsp; Algoritma: K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)

#  AUTO-RUN
auto_trigger = (st.session_state.auto_run_done is None)
should_run_builtin = run_btn or auto_trigger

#  RUN CLUSTERING DATASET BAWAAN 
auto_trigger = (st.session_state.auto_run_done is None)
should_run = run_btn or auto_trigger

#  RUN CLUSTERING (LOGIKA GABUNGAN) 
if should_run:
    # 1. CEK FILE UPLOAD
    if uploaded_xlsx is not None:
        with st.spinner("⏳ Memproses dataset kustom yang diunggah…"):
            raw = load_uploaded_dataset(uploaded_xlsx.read(), uploaded_xlsx.name)
            st.session_state.raw_stats = None # Reset statistik bawaan
            
    else:
        with st.spinner("⏳ Memuat dataset bawaan (Indore OLA)…"):
            raw = load_builtin_dataset()
            
            # Hitung statistik hanya untuk dataset bawaan
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

    # 2. PROSES DATA
    df_base = aggregate_features(raw)
    
    batas_sampel = len(df_base) if uploaded_xlsx is not None else n_sample
    df_s, df_enc, df_scaled, scaler = preprocess_for_model(df_base, batas_sampel)

    # 3. K-MEANS
    with st.spinner("⏳ Mencari K optimal dan membuat klaster…"):
        k_clusters, best_score = cari_k_optimal(df_scaled)
        km = run_kmeans(df_scaled, k_clusters)
        labels = km.labels_

    # 4. SIMPAN HASIL KE SESSION STATE
    df_result = attach_labels(df_s, labels, k_clusters)
    st.session_state.result       = df_result
    st.session_state.model_km     = km
    st.session_state.scaler       = scaler
    st.session_state.df_enc_cols  = list(df_enc.columns)
    st.session_state.k_clusters   = k_clusters
    st.session_state.best_score   = best_score
    st.session_state.auto_run_done = True

    # 5. SUKSES
    sumber_data = "Kustom" if uploaded_xlsx is not None else "Bawaan (Indore OLA)"
    st.success(
        f"Clustering Dataset {sumber_data} selesai — K optimal = **{k_clusters}** "
        f"(Silhouette Score = **{best_score:.4f}**)"
    )

#  RUN CLUSTERING DATASET KUSTOM 
if run_custom_btn and uploaded_xlsx is not None:
    with st.spinner("⏳ Memproses dataset kustom…"):
        raw_c = load_uploaded_dataset(uploaded_xlsx.read(), uploaded_xlsx.name)
        df_base_c = aggregate_features(raw_c)
        df_s_c, df_enc_c, df_scaled_c, scaler_c = preprocess_for_model(df_base_c, n_sample=len(df_base_c))

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
        st.session_state.result       = df_result_c
        st.session_state.model_km     = km_c
        st.session_state.scaler       = scaler_c
        st.session_state.df_enc_cols  = list(df_enc_c.columns)
        st.session_state.k_clusters   = k_clusters_c
        st.session_state.best_score   = best_score_c

    st.success(
        f"Clustering dataset kustom selesai — K = **{k_clusters_c}** "
        f"(Silhouette Score = **{best_score_c:.4f}**)"
    )

#  DISPLAY RESULT 
df_result = st.session_state.result

if df_result is not None:
    km       = st.session_state.model_km
    scaler   = st.session_state.scaler
    enc_cols = st.session_state.df_enc_cols
    k_val    = st.session_state.k_clusters
    b_score  = st.session_state.best_score
    raw_s    = st.session_state.raw_stats
    urutan   = [f"Klaster {i+1}" for i in range(k_val)]
    colors   = [PALETTE[i % len(PALETTE)] for i in range(k_val)]

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Ringkasan", "📈 Visualisasi", "📄 Evaluasi Model", "📋 Data Hasil"
    ])

    #  TAB 1: RINGKASAN 
    with tab1:
        # Info dataset mentah
        if raw_s:
            st.markdown('<div class="section-title">📦 Statistik Dataset Indore OLA</div>',
                        unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            for col, label, val in [
                (r1, "Total Rekaman",       f"{raw_s['total']:,}"),
                (r2, "Perjalanan Sukses",   f"{raw_s['success']:,}"),
                (r3, "Dibatalkan Customer", f"{raw_s['cancel_cust']:,}"),
                (r4, "Dibatalkan Driver",   f"{raw_s['cancel_drv']:,}"),
            ]:
                col.markdown(f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">📌 Statistik Hasil Clustering</div>',
                    unsafe_allow_html=True)

        total    = len(df_result)
        avg_frek = df_result["Frekuensi_Perjalanan"].mean()
        n_klaster= df_result["Klaster"].nunique()
        tujuan_dom = df_result["Tujuan"].mode()[0]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, val in [
            (c1, "Pelanggan Dianalisis", f"{total:,}"),
            (c2, "Rata-rata Frekuensi",  f"{avg_frek:.1f}x"),
            (c3, "Jumlah Klaster",       str(n_klaster)),
            (c4, "Tujuan Dominan",       tujuan_dom[:18]),
        ]:
            col.markdown(f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">🧩 Profil Setiap Klaster</div>',
                    unsafe_allow_html=True)

        profil = df_result.groupby("Klaster").agg(
            Jumlah_Pelanggan=("Frekuensi_Perjalanan", "count"),
            Rata_Rata_Frekuensi=("Frekuensi_Perjalanan", "mean"),
            Tujuan_Dominan=("Tujuan", lambda x: x.mode()[0]),
        ).reset_index()
        profil["Rata_Rata_Frekuensi"] = profil["Rata_Rata_Frekuensi"].round(2)
        profil["Proporsi (%)"] = (profil["Jumlah_Pelanggan"] / total * 100).round(1)

        emoji_map = {
            "Klaster 1": "🔵", "Klaster 2": "🟡",
            "Klaster 3": "🟢", "Klaster 4": "🔴", "Klaster 5": "🟣"
        }
        profil_display = profil.copy()
        profil_display["Klaster"] = profil_display["Klaster"].map(
            lambda x: f"{emoji_map.get(x, '⚪')} {x}"
        )
        st.dataframe(profil_display, use_container_width=True, hide_index=True)

        st.markdown('<div class="section-title">💡 Insight Otomatis</div>',
                    unsafe_allow_html=True)
        emojis = ["🔵", "🟡", "🟢", "🔴", "🟣"]
        for i, row in profil.iterrows():
            em = emojis[i % len(emojis)]
            tipe = ""
            if row["Rata_Rata_Frekuensi"] == profil["Rata_Rata_Frekuensi"].max():
                tipe = " — **Pengguna Paling Aktif**"
            elif row["Rata_Rata_Frekuensi"] == profil["Rata_Rata_Frekuensi"].min():
                tipe = " — **Pengguna Jarang**"
            st.info(
                f"{em} **{row['Klaster']}** — {row['Jumlah_Pelanggan']} pelanggan "
                f"({row['Proporsi (%)']}%) · rata-rata perjalanan **{row['Rata_Rata_Frekuensi']}x** "
                f"· dominan tujuan **{row['Tujuan_Dominan']}**{tipe}."
            )

    #  TAB 2: VISUALISASI 
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-title">Jumlah Pelanggan per Klaster</div>',
                        unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df_result["Klaster"].value_counts().reindex(urutan)
            bars = ax.bar(urutan, counts.values, color=colors, edgecolor="white", linewidth=1.5)
            for bar, c in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        str(c), ha="center", va="bottom", fontweight="bold", fontsize=11)
            ax.set_xlabel("Klaster"); ax.set_ylabel("Jumlah Pelanggan")
            ax.set_title("Distribusi Anggota Klaster", fontweight="bold")
            ax.grid(axis="y", alpha=0.3); ax.spines[["top","right"]].set_visible(False)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_b:
            st.markdown('<div class="section-title">Rata-rata Frekuensi per Klaster</div>',
                        unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            profil_raw = df_result.groupby("Klaster")["Frekuensi_Perjalanan"].mean().reindex(urutan)
            bars2 = ax2.bar(urutan, profil_raw.values, color=colors, edgecolor="white", linewidth=1.5)
            for bar, v in zip(bars2, profil_raw.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{v:.1f}", ha="center", va="bottom", fontweight="bold")
            ax2.set_xlabel("Klaster"); ax2.set_ylabel("Rata-rata Frekuensi")
            ax2.set_title("Rata-rata Frekuensi Perjalanan", fontweight="bold")
            ax2.grid(axis="y", alpha=0.3); ax2.spines[["top","right"]].set_visible(False)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.markdown('<div class="section-title">Heatmap Pola Tujuan Perjalanan per Klaster</div>',
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(11, 3.5))
        ct = pd.crosstab(df_result["Klaster"], df_result["Tujuan"])
        sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax3)
        ax3.set_title("Heatmap: Distribusi Tujuan per Klaster", fontweight="bold")
        ax3.set_ylabel("Klaster"); ax3.set_xlabel("Tujuan Aktivitas")
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div class="section-title">Pie Chart Komposisi Klaster</div>',
                        unsafe_allow_html=True)
            fig4, ax4 = plt.subplots(figsize=(5, 5))
            sizes = [df_result[df_result["Klaster"] == kl].shape[0] for kl in urutan]
            wedges, texts, autotexts = ax4.pie(
                sizes, labels=urutan, colors=colors,
                autopct="%1.1f%%", startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=2)
            )
            for at in autotexts:
                at.set_fontweight("bold")
            ax4.set_title("Komposisi Klaster", fontweight="bold")
            st.pyplot(fig4, use_container_width=True)
            plt.close()

        with col_d:
            st.markdown('<div class="section-title">Distribusi Frekuensi (Boxplot)</div>',
                        unsafe_allow_html=True)
            fig5, ax5 = plt.subplots(figsize=(5, 5))
            data_box = [
                df_result[df_result["Klaster"] == kl]["Frekuensi_Perjalanan"].values
                for kl in urutan
            ]
            bp = ax5.boxplot(data_box, labels=urutan, patch_artist=True)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax5.set_xlabel("Klaster"); ax5.set_ylabel("Frekuensi Perjalanan")
            ax5.set_title("Sebaran Frekuensi per Klaster", fontweight="bold")
            ax5.grid(axis="y", alpha=0.3); ax5.spines[["top","right"]].set_visible(False)
            st.pyplot(fig5, use_container_width=True)
            plt.close()

    #  TAB 3: EVALUASI 
    with tab3:
        st.markdown('<div class="section-title">Metode Elbow & Silhouette Score</div>',
                    unsafe_allow_html=True)

        with st.spinner("Menghitung skor untuk semua nilai K…"):
            df_tmp = df_result.reset_index()[["Customer Id", "Frekuensi_Perjalanan", "Tujuan"]]
            _, _, df_scaled_eval, _ = preprocess_for_model(df_tmp, n_sample=min(500, len(df_tmp)))

            inertia_list, sil_list = [], []
            K_range = range(2, 11)
            for k in K_range:
                km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
                km_tmp.fit(df_scaled_eval)
                inertia_list.append(km_tmp.inertia_)
                sil_list.append(silhouette_score(df_scaled_eval, km_tmp.labels_))

        fig6, (ax6, ax7) = plt.subplots(1, 2, figsize=(12, 4))

        ax6.plot(list(K_range), inertia_list, "bo--", linewidth=2, markersize=7)
        ax6.axvline(x=k_val, color="red", linestyle=":", alpha=0.8,
                    label=f"K optimal = {k_val}")
        ax6.set_title("Elbow Method (Inertia)", fontweight="bold")
        ax6.set_xlabel("Jumlah Klaster (K)"); ax6.set_ylabel("Inertia")
        ax6.legend(); ax6.grid(alpha=0.3)
        ax6.spines[["top","right"]].set_visible(False)

        ax7.plot(list(K_range), sil_list, "gs-", linewidth=2, markersize=7)
        ax7.axvline(x=k_val, color="red", linestyle=":", alpha=0.8,
                    label=f"K optimal = {k_val}")
        ax7.set_title("Silhouette Score per K", fontweight="bold")
        ax7.set_xlabel("Jumlah Klaster (K)"); ax7.set_ylabel("Silhouette Score")
        ax7.legend(); ax7.grid(alpha=0.3)
        ax7.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

        cur_sil = sil_list[k_val - 2]
        cur_ine = inertia_list[k_val - 2]

        e1, e2, e3 = st.columns(3)
        e1.markdown(f"""<div class="metric-card">
            <div class="label">K Optimal</div>
            <div class="value">{k_val}</div>
        </div>""", unsafe_allow_html=True)
        e2.markdown(f"""<div class="metric-card">
            <div class="label">Silhouette Score</div>
            <div class="value">{cur_sil:.4f}</div>
        </div>""", unsafe_allow_html=True)
        e3.markdown(f"""<div class="metric-card">
            <div class="label">Inertia</div>
            <div class="value">{cur_ine:,.0f}</div>
        </div>""", unsafe_allow_html=True)

        quality = (
            "Sangat Baik ✅" if cur_sil > 0.5
            else ("Cukup Baik ⚠️" if cur_sil > 0.3 else "Kurang Baik ❌")
        )
        st.info(
            f"**Interpretasi:** Silhouette Score = **{cur_sil:.4f}** "
            f"→ Kualitas Clustering: **{quality}**\n\n"
            f"Nilai Silhouette mendekati 1 berarti klaster sangat terpisah dengan baik. "
            f"Nilai > 0.5 dianggap clustering berkualitas baik."
        )

        # Tabel skor semua K
        st.markdown('<div class="section-title">Perbandingan Skor Semua Nilai K</div>',
                    unsafe_allow_html=True)
        df_eval = pd.DataFrame({
            "K": list(K_range),
            "Inertia": [f"{v:,.0f}" for v in inertia_list],
            "Silhouette Score": [f"{v:.4f}" for v in sil_list],
            "Status": ["✅ Optimal" if k == k_val else "" for k in K_range]
        })
        st.dataframe(df_eval, use_container_width=True, hide_index=True)

    #  TAB 4: DATA 
    with tab4:
        st.markdown('<div class="section-title">Tabel Hasil Clustering per Pelanggan</div>',
                    unsafe_allow_html=True)

        filter_klaster = st.multiselect(
            "Filter Klaster",
            options=df_result["Klaster"].unique().tolist(),
            default=df_result["Klaster"].unique().tolist()
        )
        filter_tujuan = st.multiselect(
            "Filter Tujuan",
            options=df_result["Tujuan"].unique().tolist(),
            default=df_result["Tujuan"].unique().tolist()
        )
        df_show = df_result[
            df_result["Klaster"].isin(filter_klaster) &
            df_result["Tujuan"].isin(filter_tujuan)
        ].reset_index()

        st.caption(f"Menampilkan **{len(df_show):,}** dari **{len(df_result):,}** pelanggan")
        st.dataframe(df_show, use_container_width=True, hide_index=True)

        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Hasil sebagai CSV",
            data=csv,
            file_name="hasil_clustering_indore_ola.csv",
            mime="text/csv",
            use_container_width=True,
        )

#  FOOTER 
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:0.78rem'>"
    "Sistem Clustering Pola Konsumsi Transportasi Online &nbsp;|&nbsp; "
    "K-Means Clustering &nbsp;|&nbsp; Dataset: Indore OLA 2025"
    "</div>",
    unsafe_allow_html=True,
)