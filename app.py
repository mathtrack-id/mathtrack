import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("📊 Math Recovery System")
st.write("Analisis Learning Loss Matematika Siswa (K-Means Clustering)")

# Upload file
uploaded_file = st.file_uploader("Upload Data Siswa (Excel)", type=["xlsx"])

if uploaded_file:
    data = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Awal")
    st.write(data)

    # ===== K-MEANS CLUSTERING =====
    X = data[['Pecahan', 'Aljabar']]

    kmeans = KMeans(n_clusters=3, random_state=0)
    data['Cluster'] = kmeans.fit_predict(X)

    # ===== INTERPRETASI CLUSTER =====
    def interpretasi(cluster):
        if cluster == 0:
            return "Kelompok 1"
        elif cluster == 1:
            return "Kelompok 2"
        else:
            return "Kelompok 3"

    data['Kategori'] = data['Cluster'].apply(interpretasi)

    # ===== OUTPUT =====
    st.subheader("📊 Hasil Clustering")
    st.write(data)

    # ===== GRAFIK =====
    st.subheader("📈 Distribusi Cluster")
    grafik = data['Kategori'].value_counts()

    fig, ax = plt.subplots()
    grafik.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # ===== REKOMENDASI =====
    st.subheader("🎯 Rekomendasi Pembelajaran")

    def rekomendasi(kategori):
        if kategori == "Kelompok 1":
            return "Fokus pada penguatan pecahan"
        elif kategori == "Kelompok 2":
            return "Fokus pada penguatan aljabar"
        else:
            return "Remedial menyeluruh"

    data['Rekomendasi'] = data['Kategori'].apply(rekomendasi)

    st.write(data[['Nama', 'Kategori', 'Rekomendasi']])

    # ===== ANALISIS KELAS =====
st.subheader("📊 Analisis Kelas")

total = len(data)
cluster_counts = data['Kategori'].value_counts()

for kategori, jumlah in cluster_counts.items():
    persen = (jumlah / total) * 100
    st.write(f"{kategori}: {jumlah} siswa ({persen:.1f}%)")

# ===== IDENTIFIKASI DOMINAN =====
dominant = cluster_counts.idxmax()

st.subheader("🎯 Kesimpulan Utama")
st.success(f"Mayoritas siswa berada pada kategori: {dominant}")

# ===== PRIORITAS PEMBELAJARAN =====
st.subheader("📌 Prioritas Pembelajaran")

if dominant == "Kelompok 1":
    st.write("Fokus utama: Penguatan konsep PECaHAN")
    st.write("Strategi: Latihan bertahap, visualisasi, dan konteks sehari-hari")
elif dominant == "Kelompok 2":
    st.write("Fokus utama: Penguatan konsep ALJABAR")
    st.write("Strategi: Pemahaman variabel, pola, dan soal kontekstual")
else:
    st.write("Fokus utama: Remedial MENYELURUH")
    st.write("Strategi: Pengulangan konsep dasar dan pendampingan intensif")

# ===== SARAN UNTUK GURU =====
st.subheader("👩‍🏫 Rekomendasi untuk Guru")

st.write("• Gunakan pembelajaran diferensiasi berdasarkan hasil clustering")
st.write("• Kelompokkan siswa sesuai kategori untuk pembelajaran targeted")
st.write("• Berikan latihan spesifik sesuai kelemahan siswa")
st.write("• Lakukan evaluasi berkala untuk melihat perkembangan")

# ===== INSIGHT TAMBAHAN =====
st.subheader("💡 Insight Sistem")

if len(cluster_counts) > 1:
    st.write("Terdapat variasi kemampuan siswa, diperlukan strategi pembelajaran adaptif")
else:
    st.write("Kemampuan siswa relatif homogen")

st.info("Sistem ini membantu guru dalam pengambilan keputusan berbasis data (data-driven decision making)")