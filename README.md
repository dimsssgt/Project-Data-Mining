# Sistem Clustering Pola Konsumsi Pengguna Transportasi Online
> Dataset: Indore OLA | Algoritma: K-Means Clustering | Dibuat dengan Streamlit

---

## Deskripsi

Aplikasi ini menganalisis pola perjalanan pengguna layanan transportasi online (OLA) di kota Indore, India. Menggunakan algoritma **K-Means Clustering**, aplikasi mengelompokkan pelanggan berdasarkan frekuensi perjalanan dan tujuan aktivitas mereka — sehingga bisa diketahui siapa pengguna aktif, pengguna jarang, dan ke mana mereka biasa pergi.

---

## Fitur

- Clustering otomatis dengan pencarian K optimal (Silhouette Score)
- Mendukung dataset bawaan maupun dataset kustom (upload file)
- Mendukung upload model `.pkl` untuk prediksi dengan model yang sudah ada
- Visualisasi lengkap: bar chart, heatmap, pie chart, boxplot
- Evaluasi model: Elbow Method & Silhouette Score
- Filter dan download hasil clustering dalam format CSV

---

## Struktur Proyek

```
ola_app/
├── app.py                              ← Entry point aplikasi
├── config.py                           ← Konstanta dan pemetaan lokasi
├── styles.py                           ← CSS kustom untuk tampilan
├── data_loader.py                      ← Load dan preprocessing dataset
├── model.py                            ← Logika K-Means Clustering
├── sidebar.py                          ← Komponen UI sidebar
├── clustering.py                       ← Orkestrasi pipeline clustering
├── indore_ola_dataset_readable.xlsx    ← Dataset bawaan
├── requirements.txt                    ← Daftar dependensi
└── tabs/
    ├── __init__.py
    ├── ringkasan.py                    ← Tab 1: Ringkasan & insight
    ├── visualisasi.py                  ← Tab 2: Grafik & visualisasi
    ├── evaluasi.py                     ← Tab 3: Evaluasi model
    └── data.py                         ← Tab 4: Tabel & download CSV
```

---

## Cara Menjalankan

**1. Clone atau download proyek ini**

```bash
git clone https://github.com/username/ola-clustering.git
cd ola-clustering
```

**2. Install dependensi**

```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**

```bash
streamlit run app.py
```

**4. Buka browser**

Aplikasi akan otomatis terbuka di `http://localhost:8501`

---

## Cara Pakai

### Dataset Bawaan
Klik tombol **"Jalankan Clustering"** di sidebar — aplikasi langsung memproses dataset Indore OLA bawaan.

### Dataset Kustom
1. Buka ekspander **"Upload Dataset Kustom"** di sidebar
2. Upload file `.xlsx` atau `.csv`
3. (Opsional) Upload model `.pkl` jika ingin menggunakan model yang sudah dilatih sebelumnya
4. Centang **"Aktifkan dataset kustom"**
5. Klik **"Jalankan dengan Dataset Kustom"**

### Format Dataset Kustom
Dataset yang di-upload harus memiliki kolom berikut:

| Kolom | Keterangan |
|---|---|
| `Customer Id` | ID unik pelanggan |
| `Booking Status` | Status perjalanan (`Success`, `Canceled by Customer`, dll.) |
| `Drop Location` | Lokasi tujuan perjalanan |

---

## Dependensi

| Library | Kegunaan |
|---|---|
| `streamlit` | Framework UI aplikasi web |
| `pandas` | Manipulasi dan analisis data |
| `scikit-learn` | Algoritma K-Means dan preprocessing |
| `matplotlib` | Plotting grafik |
| `seaborn` | Heatmap dan visualisasi statistik |
| `joblib` | Load/save model `.pkl` |
| `openpyxl` | Baca file `.xlsx` |

---

## Hasil Clustering

Setiap pelanggan dikelompokkan ke dalam klaster berdasarkan:
- **Frekuensi perjalanan** — seberapa sering mereka menggunakan layanan
- **Tujuan dominan** — kategori aktivitas yang paling sering dituju (Kuliah, Kerja, Belanja, dll.)

Hasil bisa diunduh langsung sebagai file `.csv` dari Tab **Data Hasil**.
