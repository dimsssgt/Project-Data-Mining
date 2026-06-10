import streamlit as st


def render_sidebar() -> dict:
    """
    Render sidebar konfigurasi dan kembalikan dict berisi input user:
      - run_btn: bool
      - run_custom_btn: bool
      - uploaded_xlsx: UploadedFile | None
      - uploaded_pkl: UploadedFile | None
      - n_sample: int
    """
    with st.sidebar:
        st.markdown("## Konfigurasi")
        st.markdown("---")

        n_sample = 1000

        st.markdown(" ")
        with st.expander("📁 Upload Dataset Kustom (Opsional)", expanded=False):
            st.caption("Gunakan fitur ini jika ingin mengganti dataset dengan file lain.")
            uploaded_xlsx = st.file_uploader(
                "Upload file dataset (xlsx / csv)",
                type=["xlsx", "csv"],
                key="custom_upload",
            )
            uploaded_pkl = st.file_uploader(
                "Upload model PKL (opsional)",
                type=["pkl"],
                key="custom_pkl",
            )
            use_custom = st.checkbox("Aktifkan dataset kustom", value=False)

            if use_custom and uploaded_xlsx:
                run_custom_btn = st.button(
                    "▶ Jalankan dengan Dataset Kustom",
                    use_container_width=True,
                )
            else:
                run_custom_btn = False
                if use_custom:
                    st.info("Upload file dataset terlebih dahulu.")

        run_btn = st.button(
            "Jalankan Clustering",
            use_container_width=True,
            type="primary",
        )

    return {
        "run_btn": run_btn,
        "run_custom_btn": run_custom_btn,
        "uploaded_xlsx": uploaded_xlsx,
        "uploaded_pkl": uploaded_pkl,
        "n_sample": n_sample,
    }
