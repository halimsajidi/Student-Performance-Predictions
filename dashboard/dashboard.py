import streamlit as st
import pandas as pd
import requests
from joblib import load
import os
import matplotlib.pyplot as plt
import seaborn as sns

# URL untuk file model di GitHub
model_url = "https://raw.githubusercontent.com/halimsajidi/Student-Performance-Predictions/main/dashboard/XGBoost_model.joblib"
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

# Judul aplikasi dengan deskripsi
st.title("🎓 Student Performance Prediction App")
st.write("""
Welcome to the **Student Performance Prediction App**!  
This tool uses machine learning to predict student performance based on your input data.

### How to use:
1. Upload a CSV file containing student data (features required for the model).
2. Click on **Predict** to view the predictions.
3. You will see a visual representation of the predictions.

For more details on the input format, refer to the sample CSV file [here](https://docs.google.com/spreadsheets/d/1z3ci7RPl2fuQ5xjPaVk2hZkRXxpeiVEpGpKDuQELCr4/edit?usp=sharing).
""")

# File uploader untuk mengunggah CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Membaca file CSV yang diunggah
    df_prediksi = pd.read_csv(uploaded_file)
    
    # Menampilkan preview data dengan judul dan style
    st.subheader("📋 Data Preview")
    st.dataframe(df_prediksi.head())  # Show first 5 rows for preview

    # Tambahkan deskripsi tentang kolom
    st.write("Make sure the columns match the expected input for the model.")

    # Tombol untuk melakukan prediksi
    if st.button("Predict"):
        # Memastikan data sesuai untuk model (misal preprocessing)
        
        # Melakukan prediksi
        predictions = loaded_model.predict(df_prediksi)
        
        # Menampilkan hasil prediksi
        st.subheader("📊 Predictions")
        st.write(predictions)
        
        # Visualisasi prediksi (misalnya dengan Seaborn atau Matplotlib)
        st.subheader("📈 Prediction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=predictions, ax=ax)
        ax.set_title('Distribution of Predictions')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
        st.pyplot(fig)

# Footer
st.markdown("""
---
For any questions or issues, please contact [Support](mailto:halimsajidi14@gmail.com).
""")
