import streamlit as st
import pandas as pd
import requests
from joblib import load
import os

# URL untuk file model di GitHub
model_url = "https://github.com/halimsajidi/Student-Performance-Predictions/blob/main/dashboard/XGBoost_model.joblib"
model_filename = "XGBoost_model.joblib"

# Fungsi untuk mengunduh model dari GitHub
def download_model(url, filename):
    if not os.path.isfile(filename):
        response = requests.get(url)
        with open(filename, 'wb') as file:
            file.write(response.content)

# Unduh model jika belum ada
download_model(model_url, model_filename)

# Memuat model dari file
loaded_model = load(model_filename)

# Judul aplikasi
st.title("Student Performance Prediction App")

# File uploader untuk mengunggah CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Membaca file CSV yang diunggah
    df_prediksi = pd.read_csv(uploaded_file)
    
    # Menampilkan preview data
    st.write("Data preview:")
    st.write(df_prediksi)

    # Tombol untuk melakukan prediksi
    if st.button("Predict"):
        # Memastikan data sesuai untuk model
        # Lakukan preprocessing jika diperlukan, seperti menghapus kolom yang tidak perlu
        # Melakukan prediksi
        predictions = loaded_model.predict(df_prediksi)

        # Menampilkan hasil prediksi
        st.write("Predictions:")
        st.write(predictions)
