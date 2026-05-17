import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    "statistik untuk menentukan tingkat "
    "learning loss."
)

# =====================================
# PETUNJUK INPUT
# =====================================

st.subheader("📝 Petunjuk Input Data")

st.markdown("""
### Format File Excel
- Kolom pertama = identitas siswa  
  (bebas: Nama, Name, ID, NIS, dll)

- Kolom berikutnya = data numerik/nilai

### Contoh:

| Name | Pecahan | Aljabar | Geometri |
|---|---|---|---|
| Andi | 80 | 75 | 90 |
| Budi | 60 | 55 | 70 |

### Ketentuan:
- File harus format `.xlsx`
- Nilai harus berupa angka
- Minimal terdapat 3 data
- Minimal terdapat 2 variabel numerik
""")

# =====================================
# PARAMETER
# =====================================

jumlah_cluster = 3

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
        # VALIDASI DATA KOSONG
        # =====================================

        if df.empty:

            st.error(
                "File kosong. "
                "Silakan upload data."
            )

        else:

            # =====================================
            # DETEKSI KOLOM IDENTITAS
            # =====================================

            kolom_identitas = df.columns[0]

            # =====================================
            # DATA AWAL
            # =====================================

            st.subheader("📄 Data Awal")

            st.dataframe(df)

            # =====================================
            # AMBIL KOLOM NUMERIK
            # =====================================

            kolom_numerik = df.select_dtypes(
                include='number'
            ).columns.tolist()

            # =====================================
            # HAPUS KOLOM IDENTITAS
            # =====================================

            if kolom_identitas in kolom_numerik:

                kolom_numerik.remove(
                    kolom_identitas
                )

            # =====================================
            # VALIDASI VARIABEL
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
                # VALIDASI
                # =====================================

                if len(df) < jumlah_cluster:

                    st.error(
                        "Jumlah data terlalu sedikit. "
                        "Minimal 3 data."
                    )

                elif len(fitur) < 2:

                    st.warning(
                        "Pilih minimal "
                        "2 variabel."
                    )

                elif df[fitur].isnull().sum().sum() > 0:

                    st.error(
                        "Terdapat data kosong "
                        "pada variabel."
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

                    scaler = StandardScaler()

                    X = scaler.fit_transform(
                        df[fitur]
                    )

                    # =====================================
                    # K-MEANS
                    # =====================================

                    st.subheader(
                        "🔍 K-Means Clustering"
                    )

                    st.caption(
                        "Data dikelompokkan "
                        "berdasarkan kemiripan pola."
                    )

                    kmeans = KMeans(
                        n_clusters=jumlah_cluster,
                        random_state=0
                    )

                    df['Cluster'] = (
                        kmeans.fit_predict(X)
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
                        df['Cluster'].unique()
                    ):

                        data_cluster = df[
                            df['Cluster'] == cluster
                        ]

                        rata_fitur = (
                            data_cluster[fitur]
                            .mean()
                        )

                        tertinggi = (
                            rata_fitur.idxmax()
                        )

                        terendah = (
                            rata_fitur.idxmin()
                        )

                        karakteristik = (
                            f"Cenderung tinggi pada "
                            f"{tertinggi} dan "
                            f"rendah pada "
                            f"{terendah}"
                        )

                        karakteristik_cluster[
                            cluster
                        ] = {
                            "kuat": tertinggi,
                            "lemah": terendah
                        }

                        st.write(
                            f"**Cluster {cluster}** "
                            f"→ {karakteristik}"
                        )

                    # =====================================
                    # HITUNG RATA-RATA
                    # =====================================

                    df['Rata-rata'] = (
                        df[fitur]
                        .mean(axis=1)
                    )

                    # =====================================
                    # KATEGORI LEARNING LOSS
                    # =====================================

                    st.subheader(
                        "📘 Kategori Learning Loss"
                    )

                    st.caption(
                        "Kategori ditentukan "
                        "menggunakan mean dan "
                        "standar deviasi."
                    )

                    mean = (
                        df['Rata-rata']
                        .mean()
                    )

                    sd = (
                        df['Rata-rata']
                        .std()
                    )

                    batas_tinggi = mean + sd
                    batas_rendah = mean - sd

                    st.write(
                        f"Mean (M): {mean:.2f}"
                    )

                    st.write(
                        f"Standar Deviasi (SD): "
                        f"{sd:.2f}"
                    )

                    kategori = []

                    for nilai in df['Rata-rata']:

                        if nilai >= batas_tinggi:

                            kategori.append(
                                "Learning Loss Rendah"
                            )

                        elif nilai >= batas_rendah:

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
                    # KETERANGAN
                    # =====================================

                    st.subheader(
                        "📖 Keterangan Kategori"
                    )

                    st.markdown("""
- **Learning Loss Tinggi**  
  siswa mengalami ketertinggalan belajar yang cukup besar.

- **Learning Loss Sedang**  
  siswa mengalami sebagian kesulitan belajar dan masih perlu penguatan.

- **Learning Loss Rendah**  
  siswa memiliki capaian belajar yang relatif baik.
""")

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
                            "Tidak ada siswa "
                            "prioritas."
                        )

                    # =====================================
                    # REKOMENDASI
                    # =====================================

                    st.subheader(
                        "🎯 Rekomendasi Pembelajaran"
                    )

                    st.caption(
                        "Rekomendasi diberikan "
                        "berdasarkan hasil clustering "
                        "dan tingkat learning loss."
                    )

                    rekomendasi = []

                    for i in range(len(df)):

                        cluster = (
                            df['Cluster'][i]
                        )

                        kategori_ll = (
                            df[
                                'Kategori Learning Loss'
                            ][i]
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

                        # =========================
                        # LEARNING LOSS TINGGI
                        # =========================

                        if kategori_ll == (
                            "Learning Loss Tinggi"
                        ):

                            saran = (
                                f"Siswa memerlukan "
                                f"pendampingan belajar "
                                f"secara bertahap pada "
                                f"materi {lemah}. "
                                f"Pembelajaran dapat "
                                f"dilakukan dengan "
                                f"penguatan konsep dasar, "
                                f"contoh soal sederhana, "
                                f"penggunaan media visual "
                                f"atau interaktif, latihan "
                                f"rutin, serta evaluasi "
                                f"berkala."
                            )

                        # =========================
                        # LEARNING LOSS SEDANG
                        # =========================

                        elif kategori_ll == (
                            "Learning Loss Sedang"
                        ):

                            saran = (
                                f"Siswa memerlukan "
                                f"penguatan pemahaman "
                                f"pada materi {lemah}. "
                                f"Guru dapat memberikan "
                                f"latihan kontekstual, "
                                f"diskusi kelompok, "
                                f"umpan balik berkala, "
                                f"dan latihan pemecahan "
                                f"masalah."
                            )

                        # =========================
                        # LEARNING LOSS RENDAH
                        # =========================

                        else:

                            saran = (
                                f"Siswa memiliki "
                                f"kemampuan yang relatif "
                                f"baik terutama pada "
                                f"materi {kuat}. "
                                f"Pembelajaran dapat "
                                f"diarahkan pada "
                                f"kegiatan pengayaan "
                                f"seperti soal HOTS, "
                                f"eksplorasi strategi "
                                f"pemecahan masalah, "
                                f"proyek kecil, dan "
                                f"tutor sebaya."
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
            "Terjadi kesalahan saat "
            "memproses file."
        )

        st.exception(e)

else:

    st.info(
        "Silakan upload file Excel "
        "terlebih dahulu."
    )