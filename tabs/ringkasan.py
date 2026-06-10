import pandas as pd
import streamlit as st


def render(df_result: pd.DataFrame, k_val: int, raw_stats: dict | None) -> None:
    """Render Tab 1: Ringkasan hasil clustering."""

    # ── Statistik dataset mentah (hanya dataset bawaan) ──────────────────────
    if raw_stats:
        st.markdown(
            '<div class="section-title">📦 Statistik Dataset Indore OLA</div>',
            unsafe_allow_html=True,
        )
        r1, r2, r3, r4 = st.columns(4)
        for col, label, val in [
            (r1, "Total Rekaman",       f"{raw_stats['total']:,}"),
            (r2, "Perjalanan Sukses",   f"{raw_stats['success']:,}"),
            (r3, "Dibatalkan Customer", f"{raw_stats['cancel_cust']:,}"),
            (r4, "Dibatalkan Driver",   f"{raw_stats['cancel_drv']:,}"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Statistik hasil clustering ────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">📌 Statistik Hasil Clustering</div>',
        unsafe_allow_html=True,
    )

    total      = len(df_result)
    avg_frek   = df_result["Frekuensi_Perjalanan"].mean()
    n_klaster  = df_result["Klaster"].nunique()
    tujuan_dom = df_result["Tujuan"].mode()[0]

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Pelanggan Dianalisis", f"{total:,}"),
        (c2, "Rata-rata Frekuensi",  f"{avg_frek:.1f}x"),
        (c3, "Jumlah Klaster",       str(n_klaster)),
        (c4, "Tujuan Dominan",       tujuan_dom[:18]),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{val}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Profil klaster ────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">🧩 Profil Setiap Klaster</div>',
        unsafe_allow_html=True,
    )

    profil = df_result.groupby("Klaster").agg(
        Jumlah_Pelanggan=("Frekuensi_Perjalanan", "count"),
        Rata_Rata_Frekuensi=("Frekuensi_Perjalanan", "mean"),
        Tujuan_Dominan=("Tujuan", lambda x: x.mode()[0]),
    ).reset_index()
    profil["Rata_Rata_Frekuensi"] = profil["Rata_Rata_Frekuensi"].round(2)
    profil["Proporsi (%)"] = (profil["Jumlah_Pelanggan"] / total * 100).round(1)

    emoji_map = {
        "Klaster 1": "🔵", "Klaster 2": "🟡",
        "Klaster 3": "🟢", "Klaster 4": "🔴", "Klaster 5": "🟣",
    }
    profil_display = profil.copy()
    profil_display["Klaster"] = profil_display["Klaster"].map(
        lambda x: f"{emoji_map.get(x, '⚪')} {x}"
    )
    st.dataframe(profil_display, use_container_width=True, hide_index=True)

    # ── Insight otomatis ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-title">💡 Insight Otomatis</div>',
        unsafe_allow_html=True,
    )
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
