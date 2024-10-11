# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Proyek ini bertujuan membantu meningkatkan performa akademik siswa serta mengurangi angka putus kuliah (dropout). Dengan berkembangnya teknologi dan analisis data, proyrk ini ingin menggunakan data terkait mahasiswa yang mereka peroleh dari berbagai sumber (demografi, jalur akademik, dan faktor sosial ekonomi) untuk membangun model yang dapat memprediksi keberhasilan akademik dan dropout mahasiswa sejak awal perkuliahan. Dengan prediksi ini, perusahaan dapat membantu institusi pendidikan menawarkan solusi yang tepat sasaran kepada mahasiswa.

### Permasalahan Bisnis
 Bagaimana memprediksi mahasiswa yang berpotensi putus kuliah sejak semester pertama berdasarkan data awal?

### Cakupan Proyek
- Pengumpulan Data: Mengumpulkan data dari berbagai sumber yang berisi informasi terkait mahasiswa, termasuk jalur akademik, demografi, sosial ekonomi, dan performa akademik.
- Data Understanding: Melakukan eksplorasi data untuk memahami pola, tren, dan hubungan antar fitur dalam dataset, serta mengidentifikasi variabel-variabel yang berpotensi mempengaruhi keberhasilan akademik dan risiko dropout.
- Data Preparation: Melakukan pembersihan data, penanganan missing values, transformasi fitur, dan encoding untuk memastikan data siap digunakan dalam pengembangan model machine learning.
- Pengembangan Model: Membangun model machine learning menggunakan teknik yang sesuai untuk memprediksi risiko dropout dan keberhasilan akademik mahasiswa, serta melakukan tuning hyperparameter untuk meningkatkan performa model.
- Evaluasi: Mengukur kinerja model yang dikembangkan menggunakan metrik evaluasi yang relevan (seperti akurasi, presisi, dan recall), dan melakukan analisis lebih lanjut untuk memastikan model memenuhi kebutuhan bisnis dan akurasi yang diharapkan.

### Persiapan

Sumber data: ....

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

## Business Dashboard
Dashboard ini memberikan visualisasi untuk memantau prediksi dropout mahasiswa dan keberhasilan akademik mereka. Melalui dashboard ini, pihak institusi dapat melihat tren dropout dan mengidentifikasi kelompok mahasiswa yang membutuhkan perhatian lebih.

## Menjalankan Sistem Machine Learning
Jelaskan cara menjalankan protoype sistem machine learning yang telah dibuat. Selain itu, sertakan juga link untuk mengakses prototype tersebut.

```

```

## Conclusion
Proyek ini berhasil mengembangkan sistem prediksi untuk mengidentifikasi mahasiswa dengan risiko tinggi dropout dan yang akan sukses secara akademik pada akhir semester. Sistem ini dapat membantu institusi pendidikan dalam merencanakan intervensi dini dan menawarkan dukungan kepada mahasiswa yang membutuhkan. Model machine learning yang dikembangkan mampu memberikan hasil prediksi yang cukup akurat untuk diaplikasikan pada skala yang lebih besar.

### Rekomendasi Action Items
- Implementasi Sistem di Institusi Pendidikan: Implementasikan sistem prediksi di berbagai institusi untuk membantu dalam mengambil langkah intervensi yang lebih cepat dan tepat sasaran.
- Pemantauan Berkala: Lakukan pemantauan prediksi secara berkala dan perbarui model dengan data terbaru agar prediksi tetap akurat.
- Intervensi Dini: Institusi dapat memulai program mentoring atau bimbingan khusus untuk mahasiswa yang teridentifikasi memiliki risiko tinggi untuk putus kuliah atau penurunan performa akademik.
- Kerjasama dengan Keluarga: Berikan rekomendasi intervensi kepada keluarga mahasiswa dengan risiko tinggi berdasarkan faktor sosial ekonomi.
