import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from model import preprocess_for_model


def render(df_result: pd.DataFrame, k_val: int) -> None:
    """Render Tab 3: Evaluasi model (Elbow & Silhouette)."""
    st.markdown(
        '<div class="section-title">Metode Elbow & Silhouette Score</div>',
        unsafe_allow_html=True,
    )

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

    # ── Plot Elbow + Silhouette ───────────────────────────────────────────────
    fig6, (ax6, ax7) = plt.subplots(1, 2, figsize=(12, 4))

    ax6.plot(list(K_range), inertia_list, "bo--", linewidth=2, markersize=7)
    ax6.axvline(x=k_val, color="red", linestyle=":", alpha=0.8, label=f"K optimal = {k_val}")
    ax6.set_title("Elbow Method (Inertia)", fontweight="bold")
    ax6.set_xlabel("Jumlah Klaster (K)")
    ax6.set_ylabel("Inertia")
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.spines[["top", "right"]].set_visible(False)

    ax7.plot(list(K_range), sil_list, "gs-", linewidth=2, markersize=7)
    ax7.axvline(x=k_val, color="red", linestyle=":", alpha=0.8, label=f"K optimal = {k_val}")
    ax7.set_title("Silhouette Score per K", fontweight="bold")
    ax7.set_xlabel("Jumlah Klaster (K)")
    ax7.set_ylabel("Silhouette Score")
    ax7.legend()
    ax7.grid(alpha=0.3)
    ax7.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig6, use_container_width=True)
    plt.close()

    # ── Metric cards ──────────────────────────────────────────────────────────
    cur_sil = sil_list[k_val - 2]
    cur_ine = inertia_list[k_val - 2]

    e1, e2, e3 = st.columns(3)
    e1.markdown(
        f'<div class="metric-card"><div class="label">K Optimal</div>'
        f'<div class="value">{k_val}</div></div>',
        unsafe_allow_html=True,
    )
    e2.markdown(
        f'<div class="metric-card"><div class="label">Silhouette Score</div>'
        f'<div class="value">{cur_sil:.4f}</div></div>',
        unsafe_allow_html=True,
    )
    e3.markdown(
        f'<div class="metric-card"><div class="label">Inertia</div>'
        f'<div class="value">{cur_ine:,.0f}</div></div>',
        unsafe_allow_html=True,
    )

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

    # ── Tabel perbandingan semua K ────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">Perbandingan Skor Semua Nilai K</div>',
        unsafe_allow_html=True,
    )
    df_eval = pd.DataFrame({
        "K": list(K_range),
        "Inertia": [f"{v:,.0f}" for v in inertia_list],
        "Silhouette Score": [f"{v:.4f}" for v in sil_list],
        "Status": ["✅ Optimal" if k == k_val else "" for k in K_range],
    })
    st.dataframe(df_eval, use_container_width=True, hide_index=True)
