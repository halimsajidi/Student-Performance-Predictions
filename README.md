# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Proyek ini bertujuan membantu meningkatkan performa akademik siswa serta mengurangi angka putus kuliah (dropout). Dengan berkembangnya teknologi dan analisis data, proyrk ini ingin menggunakan data terkait mahasiswa yang mereka peroleh dari berbagai sumber (demografi, jalur akademik, dan faktor sosial ekonomi) untuk membangun model yang dapat memprediksi keberhasilan akademik dan dropout mahasiswa sejak awal perkuliahan. Dengan prediksi ini, perusahaan dapat membantu institusi pendidikan menawarkan solusi yang tepat sasaran kepada mahasiswa.

### Permasalahan Bisnis
 Bagaimana memprediksi mahasiswa yang berpotensi putus kuliah sejak semester pertama berdasarkan data awal?

### Cakupan Proyek
- **Pengumpulan Data:** Mengumpulkan data dari berbagai sumber yang berisi informasi terkait mahasiswa, termasuk jalur akademik, demografi, sosial ekonomi, dan performa akademik.
- **Data Understanding:** Melakukan eksplorasi data untuk memahami pola, tren, dan hubungan antar fitur dalam dataset, serta mengidentifikasi variabel-variabel yang berpotensi mempengaruhi keberhasilan akademik dan risiko dropout.
- Data Preparation: Melakukan pembersihan data, penanganan missing values, transformasi fitur, dan encoding untuk memastikan data siap digunakan dalam pengembangan model machine learning.
- **Pengembangan Model:** Membangun model machine learning menggunakan teknik yang sesuai untuk memprediksi risiko dropout dan keberhasilan akademik mahasiswa, serta melakukan tuning hyperparameter untuk meningkatkan performa model.
- **Evaluasi:** Mengukur kinerja model yang dikembangkan menggunakan metrik evaluasi yang relevan (seperti akurasi, presisi, dan recall), dan melakukan analisis lebih lanjut untuk memastikan model memenuhi kebutuhan bisnis dan akurasi yang diharapkan.

### Persiapan

Sumber data: [dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

Setup environment:
```bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#model machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# evaluasi model
from sklearn.metrics import confusion_matrix, recall_score, f1_score, accuracy_score, precision_score

# deploy
from joblib import dump, load
```
### Data Understanding
- **Dataset students performance:** Dataset yang digunakan adalah tentang performa murid dengan jumlah data sebanyak 4424 baris dan 37 kolom yang terbagi menjadi 36 kolom independen yang akan digunakan untuk melatih model dan 1 kolom yaitu kolom Status sebagai targetnya.
- **Handling outlier:** Pada kolom Curricular_units_1st_sem_grade dan Curricular_units_2nd_sem_grade terdapat nilai 0 pada status mahasiwa graduate, hal ini tidak masuk akal seharusnya rata-rata nilai siswa yang berstatus graduate tidak boleh 0, sehingga nilai 0 ini dianggap sebagai outlier dan harus dihapus.

### Data Visualisation

**Distribusi status mahasiswa**

![image](https://github.com/user-attachments/assets/4b28dcb4-477d-4300-a95d-fcb5b355c614)

Berdasarkan plot pie chart di atas menunjukkan bahwa proporsi murid graduate sebesar 49.9% dengan jumlah murid yang dropout cukup banyak yaitu 32.1%

**Hubungan gender dengan status mahasiwa**

![image](https://github.com/user-attachments/assets/765e0085-28bf-4d26-9e75-c298ec3696e1)

Berdasarkan plot di atas siswa laki-laki memiliki persen dropout lebih tinggi (45.1%) dibandingkan dengan perempuan (25.1%). Lalu untuk tingkat gradutenya siswa laki-laki memiliki tingkat kelulusan yang rendah (35.2%) di bandingkan perempuan (57.9%).

**Hubungan marital status dan status mahasiswa**

![image](https://github.com/user-attachments/assets/72a88392-3e7a-464a-9316-f1e946e7504f)

Dapat dilihat bahwa kebanyakan siswa masih status single dan beberapa lainnya sudah menikah atau cerai. Selain itu, siswa yang sudah menikah atau cerai lebih banyak yang dropout dibandingkan graduate.

**Status berdasarkan Usia (Age)**

![image](https://github.com/user-attachments/assets/8a367ea6-3fa2-4271-badc-4ec5e0a072c2)

Dapat dilihat siswa yang berumur antara 20 sampai 30 tahun pada saat pendaftaran cenderung dropout, dibandingkan dengan siswa yang berumur dibawah 25 tahun.

**Status vs Course**

![image](https://github.com/user-attachments/assets/44cb3e2f-a789-471e-af26-609cf721e80c)

Status siswa juga dipengaruhi oleh course yang dijalani, dapat dilihat persebaran data pada grafik diatas sangat beragam, course nursing sendiri memiliki status graduate tertinggi, lalu Management (evening attendance) dan management memiliki tingkat dropout yang tinggi. Dengan demikian, status siswa dapat dipengaruhi dengan course apa yang dipilih.

**Beasiswa dan Status**

![image](https://github.com/user-attachments/assets/193e6e6f-e4ba-4865-b3c3-7d8079ed78f5)

Berdasarkan grafik dia atas, Penerima beasiswa cenderung berstatus graduate dan tidak dropout dibandingkan dengan siswa yang tidak menerima beasiswa.

**The number of curricular units vs Status**

![image](https://github.com/user-attachments/assets/f3836ce7-d7ee-4e3b-be21-b2e748ad8ea9)

![image](https://github.com/user-attachments/assets/44ec0e28-5c1b-43b9-b3f1-01e0086b5b0d)

Dapat dilihat siswa yang mengambil curricular units lebih banyak cenderung berstatus graduate dibandingkan yang tidak. Terutama siswa yang mengambil lebih dari 20 unit.

## Business Dashboard
Dashboard ini memberikan visualisasi untuk memantau prediksi dropout mahasiswa dan keberhasilan akademik mereka. Melalui dashboard ini, pihak institusi dapat melihat tren dropout dan mengidentifikasi kelompok mahasiswa yang membutuhkan perhatian lebih. 

![image](https://github.com/user-attachments/assets/e516b1b1-adb6-44a2-ac3a-dca628f345d4)

[dashboard](https://lookerstudio.google.com/reporting/6ba29a8a-bb58-49e6-98fb-e224b4b2ece7)

## Menjalankan Sistem Machine Learning
### How to Run
1. Clone the repository or download the source code.
```bash
https://github.com/halimsajidi/Student-Performance-Predictions.git
```
2. Setup Environment
```bash
conda create --name main-ds python
conda activate main-ds
```
3. Install the required Python packages
```bash
cd dashboard
```
```bash
pip install -r requirements.txt
```
4. Run the Streamlit app using:
```bash
streamlit run app.py
```
4. Buka tautan yang disediakan oleh Streamlit untuk mengakses dasbor di browser web Anda.
5. uploade test data untuk mencoba aplikasi machine learning.

[StreamlitApp](https://student-performance-predictions-halimsajidi.streamlit.app/)

## Conclusion
Berdasarkan analisis data dan insight yang telah diperoleh, beberapa faktor kunci yang dapat memprediksi potensi mahasiswa untuk putus kuliah (dropout) sejak semester pertama adalah sebagai berikut:
- **Jenis Kelamin:** Siswa laki-laki memiliki risiko putus kuliah yang lebih tinggi dibandingkan perempuan. Ini terlihat dari tingkat dropout laki-laki sebesar 45.1% sementara perempuan hanya 25.1%. Sebaliknya, tingkat kelulusan perempuan lebih tinggi (57.9%) dibandingkan laki-laki (35.2%).
- **Status Pernikahan:** Siswa yang sudah menikah atau bercerai cenderung lebih berisiko untuk putus kuliah dibandingkan dengan siswa yang masih lajang. Hal ini bisa disebabkan oleh tanggung jawab tambahan yang mereka hadapi, seperti tanggungan keluarga.
-** Usia Pendaftaran:** Mahasiswa yang mendaftar pada usia antara 20 hingga 30 tahun memiliki kemungkinan lebih tinggi untuk dropout, dibandingkan dengan mahasiswa yang mendaftar di usia yang lebih muda (di bawah 25 tahun). Ini bisa menjadi indikasi bahwa mahasiswa yang lebih tua mungkin memiliki komitmen lain di luar perkuliahan yang mempengaruhi kinerja akademis mereka.
- **Pilihan Jurusan:** Jurusan yang diambil juga sangat mempengaruhi status mahasiswa. Jurusan seperti nursing memiliki tingkat kelulusan tertinggi, sedangkan jurusan management (evening attendance) dan management memiliki tingkat dropout yang lebih tinggi. Ini menunjukkan bahwa tingkat kesulitan atau pola pembelajaran dari masing-masing program studi mungkin mempengaruhi hasil akademis siswa.
- B**easiswa:** Mahasiswa yang menerima beasiswa lebih cenderung untuk lulus dibandingkan dengan mereka yang tidak mendapatkan beasiswa. Dukungan finansial tampaknya berperan besar dalam mempertahankan siswa agar tetap melanjutkan studi hingga lulus.
- **Jumlah Unit Kurikulum:** Siswa yang mengambil lebih banyak curricular units (di atas 20 unit) memiliki peluang yang lebih besar untuk lulus. Ini menunjukkan bahwa beban akademis yang lebih berat, jika ditangani dengan baik, bisa menjadi indikator komitmen dan keberhasilan mahasiswa.
- **Nilai Akademik:** Mahasiswa yang memiliki rata-rata nilai tinggi cenderung berstatus graduate, sementara yang memiliki nilai rendah lebih berisiko untuk dropout. Nilai akademik ini merupakan faktor penting dalam memprediksi status kelulusan mahasiswa.
- **Penggunaan Model dalam Pengambilan Keputusan:** Model XGBoost terbukti sebagai model terbaik dengan akurasi sebesar 76.9%. Model ini bisa diintegrasikan dalam sistem administrasi akademik untuk membantu institusi memonitor mahasiswa dan memberikan peringatan awal (early warning system).

### Rekomendasi Action Items
- **Intervensi Awal:** Mengidentifikasi mahasiswa yang berisiko tinggi untuk dropout sejak semester pertama, terutama berdasarkan jenis kelamin, status pernikahan, dan usia pendaftaran, sehingga intervensi seperti konseling atau bimbingan tambahan dapat diberikan.
- **Dukungan Keuangan:** Memberikan lebih banyak kesempatan beasiswa kepada siswa yang berpotensi, karena ini dapat meningkatkan peluang mereka untuk lulus.
- **Pendekatan Khusus Berdasarkan Jurusan:** Menyesuaikan program dukungan atau bimbingan akademik berdasarkan jurusan yang memiliki tingkat dropout lebih tinggi, seperti jurusan management.
- **Fokus pada Nilai Akademik:** Menyediakan program bimbingan belajar atau tutor tambahan bagi mahasiswa dengan nilai akademik rendah agar dapat meningkatkan performa akademik dan mengurangi risiko putus kuliah.

Dengan memahami faktor-faktor yang mempengaruhi tingkat dropout, institusi pendidikan dapat mengambil langkah-langkah proaktif untuk mendukung mahasiswa dalam menyelesaikan studi mereka.
