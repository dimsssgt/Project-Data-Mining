import io

import pandas as pd
import streamlit as st

from config import DATASET_PATH, tentukan_aktivitas


@st.cache_data(show_spinner=False)
def load_builtin_dataset() -> pd.DataFrame:
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
def load_uploaded_dataset(file_bytes: bytes, file_name: str) -> pd.DataFrame:
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


def aggregate_features(raw: pd.DataFrame) -> pd.DataFrame:
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
