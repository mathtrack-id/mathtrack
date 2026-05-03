import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="MathTrack", layout="centered")

st.title("📊 MathTrack")
st.caption("Analisis Learning Loss Matematika Berbasis Data Mining")

uploaded_file = st.file_uploader("Upload Data Siswa (Excel)", type=["xlsx"])

if uploaded_file is not None:

    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")
    st.write(df)

    # ===== CLUSTERING =====
    X = df[['Pecahan', 'Aljabar']]

    kmeans = KMeans(n_clusters=3, random_state=0)
    df['Cluster'] = kmeans.fit_predict(X)

    mapping = {
        0: 'Kelompok 1 (Fokus Pecahan)',
        1: 'Kelompok 2 (Fokus Aljabar)',
        2: 'Kelompok 3 (Remedial)'
    }

    df['Kategori'] = df['Cluster'].map(mapping)

    # ===== HASIL =====
    st.subheader("📊 Hasil Clustering")
    st.dataframe(df)

    # ===== DISTRIBUSI =====
    st.subheader("📈 Distribusi Cluster")

    cluster_counts = df['Kategori'].value_counts()

    fig, ax = plt.subplots()
    cluster_counts.plot(kind='bar', ax=ax)
    plt.xticks(rotation=30)
    st.pyplot(fig)

    # ===== ANALISIS KELAS =====
    st.subheader("📊 Analisis Kelas")

    total = len(df)

    for kategori, jumlah in cluster_counts.items():
        persen = (jumlah / total) * 100
        st.write(f"{kategori}: {jumlah} siswa ({persen:.1f}%)")

    # ===== INSIGHT OTOMATIS (WAH BANGET) =====
    st.subheader("🧠 Insight Otomatis")

    dominan = cluster_counts.idxmax()

    if "Remedial" in dominan:
        st.error("Mayoritas siswa mengalami learning loss tinggi. Dibutuhkan intervensi segera.")
        tingkat = "TINGGI"
    elif "Pecahan" in dominan:
        st.warning("Sebagian besar siswa lemah pada pecahan.")
        tingkat = "SEDANG"
    else:
        st.success("Kemampuan siswa relatif baik, perlu penguatan ringan.")
        tingkat = "RENDAH"

    st.write(f"📌 Tingkat Learning Loss Kelas: **{tingkat}**")

    # ===== DETEKSI SISWA BERISIKO =====
    st.subheader("🚨 Siswa Prioritas")

    siswa_risiko = df[df['Kategori'].str.contains("Remedial")]

    if len(siswa_risiko) > 0:
        st.write("Siswa yang membutuhkan perhatian khusus:")
        st.dataframe(siswa_risiko)
    else:
        st.success("Tidak ada siswa dengan risiko tinggi")

    # ===== REKOMENDASI =====
    st.subheader("🎯 Rekomendasi Pembelajaran")

    rekomendasi = []

    for i in range(len(df)):
        if "Pecahan" in df['Kategori'][i]:
            rekomendasi.append("Latihan pecahan bertahap + media visual")
        elif "Aljabar" in df['Kategori'][i]:
            rekomendasi.append("Pendekatan kontekstual & pemecahan masalah")
        else:
            rekomendasi.append("Remedial intensif & pendampingan")

    df['Rekomendasi'] = rekomendasi

    st.dataframe(df[['Nama', 'Kategori', 'Rekomendasi']])

    # ===== DOWNLOAD (WAH BANGET) =====
    st.subheader("⬇️ Download Hasil")

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download hasil analisis",
        data=csv,
        file_name="hasil_mathtrack.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload file Excel terlebih dahulu")
