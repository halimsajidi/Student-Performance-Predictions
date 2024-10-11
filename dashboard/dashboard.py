import streamlit as st
import pandas as pd
from joblib import load

# Memuat model dari file
loaded_model = load('XGBoost_model.joblib')

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
