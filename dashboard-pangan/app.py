import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

# Mapping bulan dari bahasa Indonesia ke format numerik
month_mapping = {
    'Januari': '01',
    'Februari': '02',
    'Maret': '03',
    'April': '04',
    'Mei': '05',
    'Juni': '06',
    'Juli': '07',
    'Agustus': '08',
    'September': '09',
    'Oktober': '10',
    'November': '11',
    'Desember': '12'
}

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

file_path = 'dataset.xlsx' 
file_path_cleaned = 'produksi-pangan-cleaned.xlsx'
raw_data = load_data(file_path)  # Data asli yang tidak akan diubah
cleaned_data1 = load_data(file_path_cleaned)  # Data asli yang tidak akan diubah

st.title("Dashboard Komoditas")

cleaned_data1['Harga'].fillna(cleaned_data1['Harga'].median(), inplace=True)
# cleaned_data1['Harga2'].fillna(cleaned_data1['Harga2'].median(), inplace=True)
cleaned_data1['Harga2'] = cleaned_data1.groupby('Komoditas')['Harga2'].transform(lambda x: x.fillna(x.median()))

# Salin data untuk digunakan dalam manipulasi
data = raw_data.copy()

# Ubah kolom Bulan dan Tahun pada salinan data
data['Bulan'] = data['Bulan'].map(month_mapping)
if 'Bulan' in data.columns and 'Tahun' in data.columns:
    data['Tanggal'] = pd.to_datetime(data['Bulan'] + ' ' + data['Tahun'].astype(str), format='%m %Y')
    
cleaned_data1['Bulan'] = cleaned_data1['Bulan'].map(month_mapping)
if 'Bulan' in cleaned_data1.columns and 'Tahun' in cleaned_data1.columns:
    cleaned_data1['Tanggal'] = pd.to_datetime(cleaned_data1['Bulan'] + ' ' + cleaned_data1['Tahun'].astype(str), format='%m %Y')

# Dataset dengan provinsi DI Yogyakarta
data_wa = data.copy()

# Dataset tanpa provinsi DI Yogyakarta, dianggap sebagai outlier
data_woa = file_path_cleaned
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Dataset", "Basic Statistics", "Visualisasi With Anomaly", "Visualisasi Without Anomaly","Model Isolation Forest","One-Class SVM", "Best Model"])

with tab1:
    st.write("### Data Preview (Raw Data):")
    st.dataframe(raw_data)
    st.write("### Data Preview (Cleaned Data):")
    st.dataframe(cleaned_data1) 

with tab2:
    st.title("Statistik Komoditas")

    komoditas = st.selectbox("Pilih Komoditas", data_wa['Komoditas'].unique(),key=3)
  
    filtered_data = data_wa[data_wa['Komoditas'] == komoditas]

    if not filtered_data.empty:
        st.write(f"### Statistik Harga untuk {komoditas}")

        harga_min = filtered_data['Harga'].min()
        harga_mean = filtered_data['Harga'].mean()
        harga_max = filtered_data['Harga'].max()

        st.write(f"**Harga Minimum:** {harga_min}")
        st.write(f"**Harga Rata-rata:** {harga_mean}")
        st.write(f"**Harga Maksimum:** {harga_max}")

        st.write("### Data Terfilter")
        st.dataframe(filtered_data)
    else:
        st.write(f"Tidak ada data untuk komoditas {komoditas}")

with tab3:
    st.title("Visualisasi Harga Komoditas (Dengan DI Yogyakarta)")
    komoditas_visual = st.selectbox("Pilih Komoditas untuk Visualisasi", data_wa['Komoditas'].unique(), key=0)
    
    visual_data = data_wa[data_wa['Komoditas'] == komoditas_visual]

    if 'Tanggal' in visual_data.columns:
        average_prices = visual_data.groupby(['Tanggal', 'Provinsi'])['Harga'].mean().reset_index()

        fig = px.line(average_prices, x='Tanggal', y='Harga', color='Provinsi',
                      title=f'Rata-Rata Harga {komoditas_visual} per Bulan per Provinsi',
                      labels={'Harga': 'Harga(K)', 'Tanggal': 'Tanggal'},
                      markers=True)
        
        st.plotly_chart(fig)
    else:
        st.error("Kolom 'Tanggal' tidak ditemukan untuk visualisasi.")

with tab4:
    st.title("Statistik Komoditas")

    # Ambil komoditas yang dipilih dari selectbox
    komoditas = st.selectbox("Pilih Komoditas", cleaned_data1['Komoditas'].unique(), key=111222)
    
    # Gunakan variabel komoditas yang dipilih untuk memfilter data
    visual_data = cleaned_data1[cleaned_data1['Komoditas'] == komoditas]

    if 'Tanggal' in visual_data.columns:
        # Hitung rata-rata harga berdasarkan Tanggal dan Provinsi
        average_prices = visual_data.groupby(['Tanggal', 'Provinsi'])['Harga'].mean().reset_index()

        # Buat plot garis dengan plotly express
        fig = px.line(average_prices, x='Tanggal', y='Harga', color='Provinsi',
                      title=f'Tren Harga {komoditas} per Bulan per Provinsi',
                      labels={'Harga': 'Harga(K)', 'Tanggal': 'Tanggal'},
                      markers=True)
        
        st.plotly_chart(fig, key=1262)
    else:
        st.error("Kolom 'Tanggal' tidak ditemukan untuk visualisasi.")
  
with tab5:
    st.write("### Anomaly Detection for Komoditas - Isolation Forest")
    selected_commodity = st.selectbox("Pilih Komoditas untuk Deteksi Anomali", cleaned_data1['Komoditas'].unique(), key=35453)
    filtered_data = cleaned_data1[cleaned_data1['Komoditas'] == selected_commodity]

    if not filtered_data.empty:
        filtered_data['Tanggal'] = pd.to_datetime(filtered_data['Tanggal'])
        filtered_data['Tahun'] = filtered_data['Tanggal'].dt.year
        filtered_data['Bulan'] = filtered_data['Tanggal'].dt.month

        # One-hot encoding for 'Provinsi'
        data_encoded = pd.get_dummies(filtered_data, columns=['Provinsi'])

        features = ['Harga', 'Tahun', 'Bulan'] + [col for col in data_encoded.columns if col.startswith('Provinsi_')]
        X = data_encoded[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

        def evaluate_model(y_true, y_pred):
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return precision, recall, f1

        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_train)
        y_pred_if = iso_forest.predict(X_test)

        # Assume top 10% data as anomaly for evaluation
        threshold = np.percentile(X_test[:, 0], 90)
        y_true = np.where(X_test[:, 0] > threshold, -1, 1)

        precision_if, recall_if, f1_if = evaluate_model(y_true, y_pred_if)
        st.write(f"Isolation Forest - Precision: {precision_if:.3f}, Recall: {recall_if:.3f}, F1-score: {f1_if:.3f}")

        # # One-Class SVM
        # ocsvm = OneClassSVM(kernel='rbf', nu=0.001)
        # ocsvm.fit(X_train)
        # y_pred_svm = ocsvm.predict(X_test)

        # precision_svm, recall_svm, f1_svm = evaluate_model(y_true, y_pred_svm)
        # st.write(f"One-Class SVM - Precision: {precision_svm:.3f}, Recall: {recall_svm:.3f}, F1-score: {f1_svm:.3f}")

        # Visualize Anomalies
        # best_model = iso_forest if f1_if > f1_svm else ocsvm
        anomalies = iso_forest.predict(X_scaled)
        filtered_data['is_anomaly'] = anomalies

        # st.write("### Anomali Teridentifikasi")
        # st.write(filtered_data[['Tanggal', 'Harga', 'is_anomaly']].head())

        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data['Tanggal'], filtered_data['Harga'], c=filtered_data['is_anomaly'], cmap='viridis')
        plt.title(f'Deteksi Anomali Harga untuk Komoditas')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga')
        st.pyplot(plt)
    else:
        st.write("Silahkan pilih komoditas untuk mendeteksi anomali.")

with tab6:
    st.write("### Anomaly Detection for Komoditas - One Class SVM")
    selected_commodity = st.selectbox("Pilih Komoditas untuk Deteksi Anomali", cleaned_data1['Komoditas'].unique(), key=122)
    filtered_data = cleaned_data1[cleaned_data1['Komoditas'] == selected_commodity]

    if not filtered_data.empty:
        filtered_data['Tanggal'] = pd.to_datetime(filtered_data['Tanggal'])
        filtered_data['Tahun'] = filtered_data['Tanggal'].dt.year
        filtered_data['Bulan'] = filtered_data['Tanggal'].dt.month

        # One-hot encoding for 'Provinsi'
        data_encoded = pd.get_dummies(filtered_data, columns=['Provinsi'])

        features = ['Harga', 'Tahun', 'Bulan'] + [col for col in data_encoded.columns if col.startswith('Provinsi_')]
        X = data_encoded[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

        def evaluate_model(y_true, y_pred):
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return precision, recall, f1

        # One-Class SVM
        ocsvm = OneClassSVM(kernel='rbf', nu=0.001)
        ocsvm.fit(X_train)
        y_pred_svm = ocsvm.predict(X_test)

        precision_svm, recall_svm, f1_svm = evaluate_model(y_true, y_pred_svm)
        st.write(f"One-Class SVM - Precision: {precision_svm:.3f}, Recall: {recall_svm:.3f}, F1-score: {f1_svm:.3f}")

        # Visualize Anomalies
        # best_model = iso_forest if f1_if > f1_svm else ocsvm
        anomalies = ocsvm.predict(X_scaled)
        filtered_data['is_anomaly'] = anomalies

        # st.write("### Anomali Teridentifikasi")
        # st.write(filtered_data[['Tanggal', 'Harga', 'is_anomaly']].head())

        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data['Tanggal'], filtered_data['Harga'], c=filtered_data['is_anomaly'], cmap='viridis')
        plt.title(f'Deteksi Anomali Harga untuk Komoditas')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga')
        st.pyplot(plt)
    else:
        st.write("Silahkan pilih komoditas untuk mendeteksi")
        

with tab7:
    selected_commodity = st.selectbox("Pilih Komoditas untuk Deteksi Anomali", cleaned_data1['Komoditas'].unique(), key=9090)
    filtered_data = cleaned_data1[cleaned_data1['Komoditas'] == selected_commodity]

    if not filtered_data.empty:
        filtered_data['Tanggal'] = pd.to_datetime(filtered_data['Tanggal'])
        filtered_data['Tahun'] = filtered_data['Tanggal'].dt.year
        filtered_data['Bulan'] = filtered_data['Tanggal'].dt.month

        data_encoded = pd.get_dummies(filtered_data, columns=['Provinsi'])

        features = ['Harga', 'Tahun', 'Bulan'] + [col for col in data_encoded.columns if col.startswith('Provinsi_')]
        X = data_encoded[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test = train_test_split(X_scaled, test_size=0.3, random_state=42)

        def evaluate_model(y_true, y_pred):
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            return precision, recall, f1

        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(X_train)
        y_pred_if = iso_forest.predict(X_test)

        # Assume top 10% data as anomaly for evaluation
        threshold = np.percentile(X_test[:, 0], 90)
        y_true = np.where(X_test[:, 0] > threshold, -1, 1)

        precision_if, recall_if, f1_if = evaluate_model(y_true, y_pred_if)
        st.write(f"Isolation Forest - Precision: {precision_if:.3f}, Recall: {recall_if:.3f}, F1-score: {f1_if:.3f}")

        # One-Class SVM
        ocsvm = OneClassSVM(kernel='rbf', nu=0.001)
        ocsvm.fit(X_train)
        y_pred_svm = ocsvm.predict(X_test)

        precision_svm, recall_svm, f1_svm = evaluate_model(y_true, y_pred_svm)
        st.write(f"One-Class SVM - Precision: {precision_svm:.3f}, Recall: {recall_svm:.3f}, F1-score: {f1_svm:.3f}")

        # Visualize Anomalies
        best_model = iso_forest if f1_if > f1_svm else ocsvm
        is_forest_best = True if f1_if > f1_svm else False
        
        best_model_name = "IsolationForest" if is_forest_best else "OneClassSVM"
        st.title(f"Model Terbaik: {best_model_name}")
        anomalies = best_model.predict(X_scaled)
        filtered_data['is_anomaly'] = anomalies

        # st.write("# Anomali Teridentifikasi")
        # st.write(filtered_data[['Tanggal', 'Harga', 'is_anomaly']].head())

        plt.figure(figsize=(10, 6))
        plt.scatter(filtered_data['Tanggal'], filtered_data['Harga'], c=filtered_data['is_anomaly'], cmap='viridis')
        plt.title(f'Deteksi Anomali Harga untuk Komoditas')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga')
        st.pyplot(plt)
    else:
        st.write("Silahkan pilih komoditas untuk mendeteksi anomali.")