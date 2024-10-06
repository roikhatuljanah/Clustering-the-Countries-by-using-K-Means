import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Memuat model yang disimpan
scaler = joblib.load('scaler.joblib')
kmeans = joblib.load('kmeans_model.joblib')

# Memuat Data
data = pd.read_csv("data/Data_Negara_HELP.csv")

# Menangani Missing Values
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Normalisasi Data
X = data.drop(columns=['Negara'])
X_scaled = scaler.transform(X)

# Clustering
data['Cluster'] = kmeans.predict(X_scaled)

# Judul Dashboard
st.title("Dashboard Data Negara")

# Menampilkan Data
st.subheader("Data Negara")
st.dataframe(data)

# Menampilkan Jumlah Negara per Cluster
st.subheader("Jumlah Negara per Cluster")
cluster_counts = data['Cluster'].value_counts()
st.bar_chart(cluster_counts)

# Menampilkan Rata-rata Setiap Cluster
st.subheader("Rata-rata Setiap Cluster")
# Hanya gunakan kolom numerik untuk menghitung rata-rata
average_per_cluster = data.groupby('Cluster').mean(numeric_only=True).reset_index()
st.dataframe(average_per_cluster)

# Visualisasi Rata-rata Setiap Cluster
st.subheader("Visualisasi Rata-rata Setiap Cluster")
fig, ax = plt.subplots(figsize=(10, 6))
average_per_cluster.drop(columns='Cluster').plot(kind='bar', ax=ax)
ax.set_title('Rata-rata Setiap Fitur per Cluster')
ax.set_xlabel('Fitur')
ax.set_ylabel('Rata-rata')
plt.xticks(rotation=45)
st.pyplot(fig)

# Menampilkan Informasi tentang Negara di Setiap Cluster
st.subheader("Informasi tentang Negara di Setiap Cluster")
selected_cluster = st.selectbox("Pilih Cluster", options=data['Cluster'].unique())
countries_in_cluster = data[data['Cluster'] == selected_cluster]
st.dataframe(countries_in_cluster[['Negara', 'Kematian_anak', 'Ekspor', 'Kesehatan', 'Impor', 'Pendapatan', 'Inflasi', 'Harapan_hidup', 'Jumlah_fertiliti', 'GDPperkapita']])

# Fitur untuk memilih negara
st.subheader("Detail Negara")
selected_country = st.selectbox("Pilih Negara", options=data['Negara'])
country_info = data[data['Negara'] == selected_country]
st.dataframe(country_info)

# Visualisasi Detail Negara
st.subheader("Visualisasi Detail Negara")
# Mengambil fitur numerik untuk visualisasi
country_values = country_info.drop(columns=['Negara', 'Cluster']).T
country_values.columns = [selected_country]
fig, ax = plt.subplots(figsize=(10, 6))
country_values.plot(kind='bar', ax=ax)
ax.set_title(f'Detail Negara: {selected_country}')
ax.set_ylabel('Nilai')
plt.xticks(rotation=45)
st.pyplot(fig)

# Visualisasi Scatter Plot
st.subheader("Scatter Plot: Harapan Hidup vs Kematian Anak")
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(data['Harapan_hidup'], data['Kematian_anak'], c=data['Cluster'], cmap='viridis', alpha=0.6)
ax.set_title('Harapan Hidup vs Kematian Anak')
ax.set_xlabel('Harapan Hidup')
ax.set_ylabel('Kematian Anak')
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)
st.pyplot(fig)



# Visualisasi Pie Chart untuk Detail Negara
st.subheader("Pie Chart: Proporsi Fitur untuk Negara Terpilih")
fig, ax = plt.subplots(figsize=(8, 8))
country_values = country_info.drop(columns=['Negara', 'Cluster']).squeeze()  # Mengambil baris sebagai Series
ax.pie(country_values, labels=country_values.index, autopct='%1.1f%%', startangle=140)
ax.set_title(f'Proporsi Fitur untuk Negara: {selected_country}')
st.pyplot(fig)
