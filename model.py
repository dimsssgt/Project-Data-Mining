import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


def preprocess_for_model(
    df: pd.DataFrame, n_sample: int = 1000
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Stratified sampling, outlier removal, encoding, dan scaling."""
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
        columns=df_enc.columns,
        index=df_enc.index,
    )
    return df_s, df_enc, df_scaled, scaler


def cari_k_optimal(
    df_scaled: pd.DataFrame, k_min: int = 2, k_max: int = 10
) -> tuple[int, float]:
    """Cari K terbaik berdasarkan Silhouette Score."""
    best_k, best_score = 2, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(df_scaled)
        score = silhouette_score(df_scaled, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k, best_score


def run_kmeans(df_scaled: pd.DataFrame, k: int) -> KMeans:
    """Jalankan KMeans dengan K yang diberikan."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(df_scaled)
    return km


def attach_labels(df_s: pd.DataFrame, labels, k: int) -> pd.DataFrame:
    """Tambahkan kolom Klaster ke dataframe hasil."""
    df_out = df_s.copy()
    df_out["Klaster"] = [f"Klaster {l + 1}" for l in labels]
    return df_out
