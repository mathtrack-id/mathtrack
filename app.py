import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="MathTrack",
    layout="centered"
)

# =====================================
# HEADER
# =====================================

st.title("📊 MathTrack")

st.caption(
    "Analisis Learning Loss Berbasis Data Mining"
)

st.info(
    "MathTrack menggunakan metode "
    "K-Means Clustering untuk menemukan "
    "pola kemampuan siswa dan analisis "
    "learning loss secara otomatis."
)

# =====================================
# PETUNJUK INPUT
# =====================================

st.subheader("📝 Petunjuk Input Data")

st.markdown("""
### Format File Excel
- Kolom pertama = identitas siswa  
  (Nama, Name, ID, NIS, dll)

- Kolom berikutnya = data numerik/nilai

### Contoh:

| Name | Aljabar | Geometri | Bilangan |
|---|---|---|---|
| Andi | 80 | 75 | 90 |
| Budi | 60 | 55 | 70 |

### Ketentuan:
- File harus format `.xlsx`
- Nilai harus berupa angka
- Minimal terdapat 3 data
- Minimal terdapat 2 variabel numerik
- Sistem otomatis membersihkan data kosong
""")

# =====================================
# UPLOAD FILE
# =====================================

uploaded_file = st.file_uploader(
    "Upload File Excel",
    type=["xlsx"]
)

# =====================================
# PROSES
# =====================================

if uploaded_file is not None:

    try:

        # =====================================
        # MEMBACA DATA
        # =====================================

        df = pd.read_excel(uploaded_file)

        # =====================================
        # VALIDASI FILE KOSONG
        # =====================================

        if df.empty:

            st.error(
                "File kosong. "
                "Silakan upload data."
            )

        else:

            # =====================================
            # HAPUS KOLOM DUPLIKAT
            # =====================================

            df = df.loc[
                :,
                ~df.columns.duplicated()
            ]

            # =====================================
            # DETEKSI IDENTITAS
            # =====================================

            kolom_identitas = (
                df.columns[0]
            )

            # =====================================
            # DATA AWAL
            # =====================================

            st.subheader(
                "📄 Data Awal"
            )

            st.dataframe(df)

            # =====================================
            # AMBIL KOLOM NUMERIK
            # =====================================

            kolom_numerik = (
                df.select_dtypes(
                    include='number'
                )
                .columns
                .tolist()
            )

            # =====================================
            # HAPUS IDENTITAS
            # =====================================

            if (
                kolom_identitas
                in kolom_numerik
            ):

                kolom_numerik.remove(
                    kolom_identitas
                )

            # =====================================
            # VALIDASI
            # =====================================

            if len(kolom_numerik) < 2:

                st.error(
                    "Minimal harus terdapat "
                    "2 kolom numerik."
                )

            else:

                # =====================================
                # PILIH VARIABEL
                # =====================================

                st.subheader(
                    "📚 Pilih Variabel"
                )

                fitur = st.multiselect(
                    "Pilih variabel untuk analisis",
                    kolom_numerik,
                    default=kolom_numerik
                )

                # =====================================
                # VALIDASI FITUR
                # =====================================

                if len(fitur) < 2:

                    st.warning(
                        "Pilih minimal "
                        "2 variabel."
                    )

                else:

                    # =====================================
                    # HAPUS DATA KOSONG
                    # =====================================

                    if (
                        df[fitur]
                        .isnull()
                        .sum()
                        .sum()
                        > 0
                    ):

                        st.warning(
                            "Terdapat data kosong. "
                            "Sistem akan menghapus "
                            "baris yang memiliki "
                            "data kosong."
                        )

                        df = df.dropna(
                            subset=fitur
                        )

                        st.info(
                            f"Jumlah data setelah "
                            f"pembersihan: "
                            f"{len(df)} data"
                        )

                    # =====================================
                    # VALIDASI JUMLAH DATA
                    # =====================================

                    if len(df) < 3:

                        st.error(
                            "Jumlah data terlalu sedikit."
                        )

                    else:

                        # =====================================
                        # STANDARDISASI
                        # =====================================

                        st.subheader(
                            "⚖️ Standardisasi Data"
                        )

                        st.caption(
                            "Data dinormalisasi agar "
                            "setiap variabel memiliki "
                            "pengaruh yang seimbang."
                        )

                        scaler = (
                            StandardScaler()
                        )

                        X = scaler.fit_transform(
                            df[fitur]
                        )

                        st.success(
                            "Standardisasi berhasil."
                        )

                        # =====================================
                        # MENENTUKAN CLUSTER TERBAIK
                        # =====================================

                        st.subheader(
                            "🔍 Penentuan Cluster Terbaik"
                        )

                        st.caption(
                            "Sistem mencari jumlah "
                            "cluster terbaik "
                            "menggunakan "
                            "Silhouette Score."
                        )

                        silhouette_scores = []

                        range_cluster = (
                            range(2, 7)
                        )

                        for k in range_cluster:

                            model = KMeans(
                                n_clusters=k,
                                random_state=0
                            )

                            cluster_labels = (
                                model.fit_predict(X)
                            )

                            score = (
                                silhouette_score(
                                    X,
                                    cluster_labels
                                )
                            )

                            silhouette_scores.append(
                                score
                            )

                        # =====================================
                        # CLUSTER TERBAIK
                        # =====================================

                        jumlah_cluster = (
                            range_cluster[
                                silhouette_scores.index(
                                    max(
                                        silhouette_scores
                                    )
                                )
                            ]
                        )

                        st.success(
                            f"Jumlah cluster terbaik: "
                            f"{jumlah_cluster}"
                        )

                        # =====================================
                        # K-MEANS FINAL
                        # =====================================

                        st.subheader(
                            "🧠 K-Means Clustering"
                        )

                        kmeans = KMeans(
                            n_clusters=jumlah_cluster,
                            random_state=0
                        )

                        df['Cluster'] = (
                            kmeans.fit_predict(X)
                        )

                        st.success(
                            "Clustering berhasil."
                        )

                        # =====================================
                        # HASIL CLUSTER
                        # =====================================

                        st.subheader(
                            "📊 Hasil Clustering"
                        )

                        st.dataframe(
                            df[
                                [kolom_identitas]
                                + fitur
                                + ['Cluster']
                            ]
                        )

                        # =====================================
                        # KARAKTERISTIK CLUSTER
                        # =====================================

                        st.subheader(
                            "🧠 Karakteristik Cluster"
                        )

                        karakteristik_cluster = {}

                        for cluster in sorted(
                            df['Cluster']
                            .unique()
                        ):

                            data_cluster = df[
                                df['Cluster']
                                == cluster
                            ]

                            rata_fitur = (
                                data_cluster[
                                    fitur
                                ]
                                .mean()
                            )

                            tertinggi = (
                                rata_fitur.idxmax()
                            )

                            terendah = (
                                rata_fitur.idxmin()
                            )

                            karakteristik_cluster[
                                cluster
                            ] = {
                                "kuat": tertinggi,
                                "lemah": terendah
                            }

                            st.write(
                                f"""
                                **Cluster {cluster}**
                                - Kemampuan relatif baik pada: **{tertinggi}**
                                - Kemampuan relatif rendah pada: **{terendah}**
                                """
                            )

                        # =====================================
                        # HITUNG RATA-RATA
                        # =====================================

                        df['Rata-rata'] = (
                            df[fitur]
                            .mean(axis=1)
                        )

                        # =====================================
                        # LEARNING LOSS
                        # =====================================

                        st.subheader(
                            "📘 Analisis Learning Loss"
                        )

                        mean = (
                            df['Rata-rata']
                            .mean()
                        )

                        sd = (
                            df['Rata-rata']
                            .std()
                        )

                        batas_atas = (
                            mean + sd
                        )

                        batas_bawah = (
                            mean - sd
                        )

                        st.write(
                            f"Mean (M): "
                            f"{mean:.2f}"
                        )

                        st.write(
                            f"Standar Deviasi (SD): "
                            f"{sd:.2f}"
                        )

                        kategori = []

                        for nilai in (
                            df['Rata-rata']
                        ):

                            if nilai >= batas_atas:

                                kategori.append(
                                    "Learning Loss Rendah"
                                )

                            elif (
                                nilai >= batas_bawah
                            ):

                                kategori.append(
                                    "Learning Loss Sedang"
                                )

                            else:

                                kategori.append(
                                    "Learning Loss Tinggi"
                                )

                        df[
                            'Kategori Learning Loss'
                        ] = kategori

                        # =====================================
                        # HASIL LEARNING LOSS
                        # =====================================

                        st.subheader(
                            "📋 Hasil Learning Loss"
                        )

                        st.dataframe(
                            df[
                                [
                                    kolom_identitas,
                                    'Cluster',
                                    'Rata-rata',
                                    'Kategori Learning Loss'
                                ]
                            ]
                        )

                        # =====================================
                        # DISTRIBUSI
                        # =====================================

                        st.subheader(
                            "📈 Distribusi Learning Loss"
                        )

                        distribusi = (
                            df[
                                'Kategori Learning Loss'
                            ]
                            .value_counts()
                        )

                        fig, ax = plt.subplots()

                        distribusi.plot(
                            kind='bar',
                            ax=ax
                        )

                        plt.xticks(rotation=10)

                        st.pyplot(fig)

                        # =====================================
                        # SISWA PRIORITAS
                        # =====================================

                        st.subheader(
                            "🚨 Siswa Prioritas"
                        )

                        prioritas = df[
                            df[
                                'Kategori Learning Loss'
                            ]
                            == "Learning Loss Tinggi"
                        ]

                        if len(prioritas) > 0:

                            st.dataframe(
                                prioritas[
                                    [
                                        kolom_identitas,
                                        'Rata-rata',
                                        'Kategori Learning Loss'
                                    ]
                                ]
                            )

                        else:

                            st.success(
                                "Tidak ada siswa prioritas."
                            )

                        # =====================================
                        # REKOMENDASI
                        # =====================================

                        st.subheader(
                            "🎯 Rekomendasi Pembelajaran"
                        )

                        rekomendasi = []

                        for i in range(len(df)):

                            cluster = (
                                df['Cluster']
                                .iloc[i]
                            )

                            kategori_ll = (
                                df[
                                    'Kategori Learning Loss'
                                ]
                                .iloc[i]
                            )

                            lemah = (
                                karakteristik_cluster[
                                    cluster
                                ]['lemah']
                            )

                            kuat = (
                                karakteristik_cluster[
                                    cluster
                                ]['kuat']
                            )

                            # =====================================
                            # LEARNING LOSS TINGGI
                            # =====================================

                            if (
                                kategori_ll
                                == "Learning Loss Tinggi"
                            ):

                                saran = (
                                    f"Penguatan konsep dasar "
                                    f"pada materi {lemah}, "
                                    f"latihan bertahap, "
                                    f"pendampingan belajar, "
                                    f"dan evaluasi rutin."
                                )

                            # =====================================
                            # LEARNING LOSS SEDANG
                            # =====================================

                            elif (
                                kategori_ll
                                == "Learning Loss Sedang"
                            ):

                                saran = (
                                    f"Penguatan pemahaman "
                                    f"materi {lemah} "
                                    f"melalui latihan "
                                    f"kontekstual dan "
                                    f"diskusi kelompok."
                                )

                            # =====================================
                            # LEARNING LOSS RENDAH
                            # =====================================

                            else:

                                saran = (
                                    f"Pengayaan materi "
                                    f"{kuat}, soal HOTS, "
                                    f"dan tutor sebaya."
                                )

                            rekomendasi.append(
                                saran
                            )

                        df['Rekomendasi'] = (
                            rekomendasi
                        )

                        st.dataframe(
                            df[
                                [
                                    kolom_identitas,
                                    'Cluster',
                                    'Kategori Learning Loss',
                                    'Rekomendasi'
                                ]
                            ]
                        )

                        # =====================================
                        # DOWNLOAD
                        # =====================================

                        st.subheader(
                            "⬇️ Download Hasil"
                        )

                        csv = (
                            df.to_csv(index=False)
                            .encode('utf-8')
                        )

                        st.download_button(
                            label="Download Hasil",
                            data=csv,
                            file_name="hasil_mathtrack.csv",
                            mime="text/csv"
                        )

    except Exception as e:

        st.error(
            "Terjadi kesalahan "
            "saat memproses file."
        )

        st.exception(e)

else:

    st.info(
        "Silakan upload file Excel terlebih dahulu."
    )