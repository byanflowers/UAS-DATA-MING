# UAS-DATA-MINING
04TPLP024

ABYAN SANTOSO (231011402680)

ADNAN KAHFI BARA (231011400993)

# 🩺 Analisis Data Mining: Prediksi Penyakit Diabetes Berdasarkan Jurnal Ilmiah

LINK LAPORAN VIDIO YOUTUBE
https://youtu.be/9jwBeo1Vis4?feature=shared

## 📌 Pendahuluan

Data mining adalah proses penting dalam menganalisis data besar untuk menemukan pola dan informasi tersembunyi.  

Dalam bidang kesehatan, data mining membantu memprediksi risiko penyakit sehingga keputusan medis dapat dibuat lebih cepat dan tepat.


## 🎯 Tujuan

Menganalisis dataset diabetes menggunakan proses normalisasi data, dengan referensi dari jurnal ilmiah:  

> **"Prediksi Penyakit Diabetes Menggunakan Random Forest dan XGBoost"**  

> Jurnal Ilmiah Komputasi, Volume 23 No 1, Tahun 2023  

> [Link jurnal (Garuda Ristekdikti)](https://garuda.kemdikbud.go.id/documents/detail/3528811)


## 📂 Dataset

Dataset yang digunakan adalah [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) dari Kaggle.  

Dataset ini berisi **768 data pasien** dengan atribut:

- Pregnancies

- Glucose

- BloodPressure

- SkinThickness

- Insulin

- BMI

- DiabetesPedigreeFunction

- Age

Kolom target: `Outcome` (1 = positif diabetes, 0 = negatif diabetes).


## ⚙️ Proses Normalisasi

Untuk mempersiapkan data sebelum pemodelan, dilakukan normalisasi agar semua fitur berada di rentang **0–1**.

Menggunakan `MinMaxScaler` dari pustaka `scikit-learn`:

```python

from sklearn.preprocessing import MinMaxScaler

import pandas as pd

# Contoh: load dataset

df = pd.read_csv('diabetes.csv')

# Pilih hanya kolom numerik (tanpa target)

X = df.drop(columns=['Outcome'])

# Inisialisasi scaler

scaler = MinMaxScaler()

# Fit dan transformasi data

normalized_data = scaler.fit_transform(X)

# Ubah jadi DataFrame agar rapi

normalized_df = pd.DataFrame(normalized_data, columns=X.columns)

# Cetak hasil

print(normalized_df.head())

