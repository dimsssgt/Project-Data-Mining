import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from config import PALETTE


def render(df_result: pd.DataFrame, k_val: int) -> None:
    """Render Tab 2: Visualisasi hasil clustering."""
    urutan = [f"Klaster {i + 1}" for i in range(k_val)]
    colors = [PALETTE[i % len(PALETTE)] for i in range(k_val)]

    col_a, col_b = st.columns(2)

    # ── Bar chart: jumlah pelanggan per klaster ───────────────────────────────
    with col_a:
        st.markdown(
            '<div class="section-title">Jumlah Pelanggan per Klaster</div>',
            unsafe_allow_html=True,
        )
        fig, ax = plt.subplots(figsize=(5, 4))
        counts = df_result["Klaster"].value_counts().reindex(urutan)
        bars = ax.bar(urutan, counts.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, c in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(c), ha="center", va="bottom", fontweight="bold", fontsize=11,
            )
        ax.set_xlabel("Klaster")
        ax.set_ylabel("Jumlah Pelanggan")
        ax.set_title("Distribusi Anggota Klaster", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Bar chart: rata-rata frekuensi per klaster ────────────────────────────
    with col_b:
        st.markdown(
            '<div class="section-title">Rata-rata Frekuensi per Klaster</div>',
            unsafe_allow_html=True,
        )
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        profil_raw = df_result.groupby("Klaster")["Frekuensi_Perjalanan"].mean().reindex(urutan)
        bars2 = ax2.bar(urutan, profil_raw.values, color=colors, edgecolor="white", linewidth=1.5)
        for bar, v in zip(bars2, profil_raw.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v:.1f}", ha="center", va="bottom", fontweight="bold",
            )
        ax2.set_xlabel("Klaster")
        ax2.set_ylabel("Rata-rata Frekuensi")
        ax2.set_title("Rata-rata Frekuensi Perjalanan", fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)
        ax2.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # ── Heatmap: pola tujuan per klaster ─────────────────────────────────────
    st.markdown(
        '<div class="section-title">Heatmap Pola Tujuan Perjalanan per Klaster</div>',
        unsafe_allow_html=True,
    )
    fig3, ax3 = plt.subplots(figsize=(11, 3.5))
    ct = pd.crosstab(df_result["Klaster"], df_result["Tujuan"])
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax3)
    ax3.set_title("Heatmap: Distribusi Tujuan per Klaster", fontweight="bold")
    ax3.set_ylabel("Klaster")
    ax3.set_xlabel("Tujuan Aktivitas")
    st.pyplot(fig3, use_container_width=True)
    plt.close()

    col_c, col_d = st.columns(2)

    # ── Pie chart: komposisi klaster ──────────────────────────────────────────
    with col_c:
        st.markdown(
            '<div class="section-title">Pie Chart Komposisi Klaster</div>',
            unsafe_allow_html=True,
        )
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        sizes = [df_result[df_result["Klaster"] == kl].shape[0] for kl in urutan]
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=urutan, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2),
        )
        for at in autotexts:
            at.set_fontweight("bold")
        ax4.set_title("Komposisi Klaster", fontweight="bold")
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # ── Boxplot: distribusi frekuensi ─────────────────────────────────────────
    with col_d:
        st.markdown(
            '<div class="section-title">Distribusi Frekuensi (Boxplot)</div>',
            unsafe_allow_html=True,
        )
        fig5, ax5 = plt.subplots(figsize=(5, 5))
        data_box = [
            df_result[df_result["Klaster"] == kl]["Frekuensi_Perjalanan"].values
            for kl in urutan
        ]
        bp = ax5.boxplot(data_box, labels=urutan, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax5.set_xlabel("Klaster")
        ax5.set_ylabel("Frekuensi Perjalanan")
        ax5.set_title("Sebaran Frekuensi per Klaster", fontweight="bold")
        ax5.grid(axis="y", alpha=0.3)
        ax5.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig5, use_container_width=True)
        plt.close()
