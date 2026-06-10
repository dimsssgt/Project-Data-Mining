import pandas as pd
import streamlit as st


def render(df_result: pd.DataFrame) -> None:
    """Render Tab 4: Tabel hasil clustering beserta filter dan download."""
    st.markdown(
        '<div class="section-title">Tabel Hasil Clustering per Pelanggan</div>',
        unsafe_allow_html=True,
    )

    filter_klaster = st.multiselect(
        "Filter Klaster",
        options=df_result["Klaster"].unique().tolist(),
        default=df_result["Klaster"].unique().tolist(),
    )
    filter_tujuan = st.multiselect(
        "Filter Tujuan",
        options=df_result["Tujuan"].unique().tolist(),
        default=df_result["Tujuan"].unique().tolist(),
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
