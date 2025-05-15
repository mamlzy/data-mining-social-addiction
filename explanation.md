# Penjelasan Kode Analisis Kecanduan Media Sosial pada Mahasiswa

Notebook ini berisi analisis data dan pembuatan model prediksi tentang tingkat kecanduan media sosial pada mahasiswa. Berikut penjelasan bagian-bagian utamanya:

## 1. Pengumpulan dan Persiapan Data

- Notebook dimulai dengan mengimpor pustaka yang diperlukan seperti pandas, numpy, seaborn dan matplotlib untuk analisis data dan visualisasi.
- Dataset yang digunakan berisi informasi tentang 705 mahasiswa dengan berbagai atribut seperti umur, jenis kelamin, negara, jumlah jam penggunaan media sosial per hari, platform yang paling sering digunakan, skor kesehatan mental, dan skor kecanduan.

## 2. Eksplorasi dan Visualisasi Data

- Dilakukan visualisasi untuk memahami hubungan antara berbagai variabel dengan tingkat kecanduan media sosial.
- Ada analisis distribusi pengguna berdasarkan negara menggunakan peta choropleth.
- Dilakukan analisis penggunaan platform media sosial berdasarkan jenis kelamin, yang menunjukkan:
  - Perempuan paling banyak menggunakan Instagram (172 orang)
  - Laki-laki paling banyak menggunakan Facebook (99 orang)
- Data divisualisasikan dalam bentuk grafik batang yang membandingkan penggunaan platform berdasarkan gender.

## 3. Persiapan Data untuk Pemodelan

- Beberapa fitur yang tidak relevan dihapus dan data dibersihkan.
- Data outlier (pencilan) dihapus dengan menggunakan teknik kuantil 99%.
- Skor kecanduan (Addicted_Score) ditransformasi menggunakan logaritma (log1p) untuk membuat distribusinya lebih normal.
- Data dibagi menjadi variabel independen (X) dan variabel target (y).

## 4. Pembuatan Model Prediksi

- Model dibuat menggunakan pipeline scikit-learn yang terdiri dari:
  - Preprocessor yang menggunakan:
    - StandardScaler untuk fitur numerik (Umur, Jam Penggunaan Harian, Jam Tidur per Malam)
    - OneHotEncoder untuk fitur kategorikal (Jenis Kelamin)
  - RandomForestRegressor untuk memprediksi skor kecanduan
- Data dibagi menjadi 80% data latih dan 20% data uji.

## 5. Evaluasi Model

- Model dievaluasi menggunakan beberapa metrik:
  - MSE (Mean Squared Error): 0.0111
  - RMSE (Root Mean Squared Error): 0.1054
  - MAE (Mean Absolute Error): 0.0679
  - R² Score: 0.7602
- R² Score sekitar 0.76 menunjukkan bahwa model dapat menjelaskan sekitar 76% variasi dalam data skor kecanduan, yang merupakan hasil yang cukup baik.

## Kesimpulan

Notebook ini berhasil membuat model prediksi skor kecanduan media sosial mahasiswa dengan akurasi yang cukup baik berdasarkan faktor seperti umur, jenis kelamin, jam penggunaan harian, dan jam tidur per malam. Model ini bisa digunakan untuk mengidentifikasi mahasiswa yang berisiko mengalami kecanduan media sosial berdasarkan karakteristik tersebut.

---

_**User**_

mohon jelaskan juga per kodenya

---

_**Assistant**_

Saya akan menjelaskan kode dalam notebook ini secara detail:

### Import Library

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Kode ini mengimpor pustaka yang dibutuhkan:

- pandas: untuk manipulasi dan analisis data
- numpy: untuk operasi matematika dan array
- seaborn: untuk visualisasi data statistik
- matplotlib.pyplot: untuk membuat plot dan grafik

### Memuat dan Melihat Data

```python
# Kode membaca dataset
df = pd.read_csv("...")  # diasumsikan ada kode pembacaan data
```

Membaca dataset ke dalam DataFrame pandas.

### Analisis Distribusi Pengguna berdasarkan Negara

```python
import plotly.express as px

country_counts = df['Country'].value_counts().reset_index()
country_counts.columns = ['Country', 'Count']

# Visualize the distribution of users by country
fig = px.choropleth(
    country_counts,
    locations='Country',
    locationmode='country names',
    color='Count',
    color_continuous_scale="Greens",
    title='Distribution of Users by Country'
)
fig.show()
```

Kode ini:

1. Menghitung jumlah pengguna per negara menggunakan `value_counts()`
2. Mereset indeks dan memberi nama kolom
3. Membuat peta choropleth dengan plotly express yang menunjukkan distribusi pengguna di seluruh dunia
4. Warna hijau yang lebih gelap menunjukkan jumlah pengguna yang lebih tinggi

### Analisis Penggunaan Platform berdasarkan Jenis Kelamin

```python
drop_platforms = ['All','VKontakte', 'Snapchat', 'WeChat', 'LINE', 'KakaoTalk', 'LinkedIn']
most_used_paltform_by_gender = pd.crosstab(df["Most_Used_Platform"], df["Gender"], margins=True).drop(drop_platforms, axis=0)
sorted_platforms = most_used_paltform_by_gender.sort_values(by='All', ascending=False)

print(sorted_platforms)

top_female_platforms = most_used_paltform_by_gender['Female'].idxmax()
female_count = most_used_paltform_by_gender['Female'].max()

top_male_platforms = most_used_paltform_by_gender['Male'].idxmax()
male_count = most_used_paltform_by_gender['Male'].max()

print("")
print(f"Cewek paling banyak menggunakan: {top_female_platforms} ({female_count} orang)")
print(f"Cewek paling banyak menggunakan: {top_male_platforms} ({male_count} orang)")
```

Kode ini:

1. Membuat tabel kontingensi (crosstab) antara platform yang paling sering digunakan dan jenis kelamin
2. Mengurutkan berdasarkan jumlah total pengguna ('All')
3. Mengidentifikasi platform yang paling populer di kalangan perempuan dan laki-laki
4. Mencetak hasilnya (ada kesalahan pada baris terakhir, seharusnya "Cowok paling banyak menggunakan")

### Visualisasi Platform berdasarkan Jenis Kelamin

```python
gender_platforms = most_used_paltform_by_gender[['Female', 'Male']].sort_values(by='Female', ascending=False)

# Plot
gender_platforms.plot(kind='bar', figsize=(12, 6), color=['pink', 'skyblue'])
plt.title("Penggunaan Platform Berdasarkan Gender")
plt.xlabel("Platform")
plt.ylabel("Jumlah Pengguna")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Kode ini:

1. Mengambil data jumlah pengguna laki-laki dan perempuan untuk setiap platform
2. Membuat grafik batang yang membandingkan penggunaan platform berdasarkan gender
3. Mengatur judul, label, dan rotasi sumbu x untuk keterbacaan yang lebih baik

### Persiapan Data untuk Pemodelan

```python
q = df['Addicted_Score'].quantile(0.99)
X_Addicted_Score = df[df['Addicted_Score'] < q]
data_cleaned = X_Addicted_Score.reset_index(drop=True)
```

Kode ini:

1. Menghitung kuantil ke-99 dari skor kecanduan
2. Menghapus data pencilan (outlier) dengan menyaring nilai di atas kuantil tersebut
3. Mereset indeks DataFrame yang telah dibersihkan

### Transformasi dan Persiapan Fitur

```python
# Buatkan pre processing log dari addicted score
data_cleaned['Addicted_Score_Log'] = np.log1p(data_cleaned['Addicted_Score'])

X = data_cleaned.drop(['Addicted_Score', 'Addicted_Score_Log'], axis=1)
y = data_cleaned['Addicted_Score_Log']

# Identifikasi kolom kategorikal dan numerikal
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
```

Kode ini:

1. Menerapkan transformasi logaritmik (log1p) pada skor kecanduan untuk membuat distribusinya lebih normal
2. Memisahkan fitur (X) dan target (y)
3. Mengidentifikasi kolom kategorikal dan numerik secara otomatis

### Pembuatan Pipeline dan Model

```python
# Buat preprocessor: OneHot untuk kategori, StandardScaler untuk numerik
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Bikin pipeline dengan Random Forest Regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)
```

Kode ini:

1. Membuat transformer kolom yang menerapkan:
   - StandardScaler untuk kolom numerik (menstandarisasi data)
   - OneHotEncoder untuk kolom kategorikal (mengonversi kategori ke format numerik)
2. Membuat pipeline yang menggabungkan preprocessor dengan model Random Forest
3. Membagi data menjadi set pelatihan (80%) dan pengujian (20%)
4. Melatih model dengan data pelatihan

### Evaluasi Model

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Prediksi
y_pred = model.predict(X_test)

# Hitung metrik
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R² Score:", r2)
```

Kode ini:

1. Menggunakan model untuk membuat prediksi pada data pengujian
2. Menghitung berbagai metrik evaluasi:
   - MSE (Mean Squared Error): rata-rata kuadrat error
   - RMSE (Root Mean Squared Error): akar kuadrat dari MSE
   - MAE (Mean Absolute Error): rata-rata nilai absolut error
   - R² Score: proporsi variasi target yang dapat dijelaskan oleh model
3. Mencetak semua metrik evaluasi

R² Score 0.76 menunjukkan bahwa model ini dapat menjelaskan 76% variasi dalam skor kecanduan, yang mengindikasikan model prediksi yang cukup baik untuk kasus ini.

---
