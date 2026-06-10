import os

# ─── PATH ───────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "indore_ola_dataset_readable.xlsx")

# ─── PEMETAAN LOKASI → KATEGORI AKTIVITAS ───────────────────────────────────
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


def tentukan_aktivitas(lokasi: str) -> str:
    return TUJUAN_MAP.get(lokasi.strip(), "Lainnya")
