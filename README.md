# Proyek Akhir: Prediksi Risiko Dropout Mahasiswa - Jaya Jaya Institut

## Business Understanding

Jaya Jaya Institut ingin meningkatkan retensi dan kelulusan mahasiswa. Dropout yang terlambat teridentifikasi dapat membuat mahasiswa kehilangan kesempatan memperoleh bantuan akademik atau finansial, sekaligus menurunkan tingkat kelulusan, reputasi, dan efektivitas operasional institusi.

Hasil proyek ini digunakan oleh manajemen akademik, dosen pembimbing akademik, unit kemahasiswaan, serta unit keuangan/layanan bantuan mahasiswa. Dashboard digunakan untuk memantau kelompok dengan risiko lebih tinggi, sedangkan prototype model digunakan sebagai *early warning system* untuk memprioritaskan peninjauan kasus oleh manusia.

Tujuan bisnisnya adalah mengarahkan intervensi lebih dini dan tepat sasaran: pendampingan akademik, pengingat serta konsultasi pembayaran, bantuan beasiswa, dan outreach proaktif. Dampak yang diharapkan adalah berkurangnya dropout yang dapat dicegah, alokasi sumber daya pendampingan yang lebih baik, dan hasil belajar mahasiswa yang lebih baik.

### Permasalahan Bisnis

1. Mahasiswa berisiko dropout sulit dikenali sebelum benar-benar berhenti, sehingga institusi terlambat menawarkan bantuan.
2. Institusi perlu mengetahui kelompok mana yang memiliki tingkat dropout tinggi berdasarkan kinerja akademik, tunggakan, status pembayaran, beasiswa, serta karakteristik demografis.
3. Dosen pembimbing dan unit kemahasiswaan membutuhkan daftar prioritas yang dapat ditindaklanjuti melalui mentoring, remedial, konseling, atau bantuan finansial.
4. Unit keuangan perlu membedakan kebutuhan follow-up pembayaran biasa dengan mahasiswa yang juga memperlihatkan sinyal risiko akademik.

Prediksi dipakai sebagai rekomendasi prioritas untuk *human review*, bukan sebagai dasar pemberian sanksi atau keputusan akademik otomatis.

### Cakupan Proyek

- Menganalisis faktor yang berkaitan dengan status mahasiswa dan dropout.
- Melatih model biner untuk membedakan outcome `Dropout` dan `Graduate`.
- Menyediakan prototype Streamlit untuk mengestimasi probabilitas risiko dropout.
- Menyediakan dashboard Metabase untuk memantau status mahasiswa dan faktor risiko utama.

### Batasan

Status `Enrolled` tidak digunakan saat pelatihan model karena outcome akhirnya belum terkonfirmasi. Hasil analisis bersifat korelasional dan tidak membuktikan hubungan sebab-akibat.

### Persiapan

#### Sumber data

- [students_performance.csv](https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/main/students_performance/data.csv)
- Salinan data lokal: `data/data.csv`

#### Setup environment

Proyek dikembangkan dan diuji menggunakan **Python 3.12.14**.

```powershell
cd D:\Dicoding\submission_03
.\.venv\Scripts\Activate.ps1
cd .\a590_proyek_akhir
python -m pip install -r requirements.txt
```

## Business Dashboard

Dashboard bernama **Dashboard Analisis Risiko Dropout Mahasiswa** dibuat menggunakan **Metabase v0.46.4**. Bukti tampilan tersedia pada `nafis_fakhru-dashboard.png`, sedangkan konfigurasi dashboard tersimpan pada `metabase.db.mv.db`.

Dashboard memantau total mahasiswa, persentase dropout, distribusi status, status pembayaran, nilai semester 2, serta faktor risiko tunggakan, beasiswa, dan gender. Visualisasi persentase digunakan untuk membandingkan risiko antarkelompok secara lebih adil daripada hanya membandingkan jumlah mahasiswa.

### Menjalankan dashboard Metabase

Prasyarat: Docker Desktop sudah berjalan. Dashboard menggunakan data `students_performance`; untuk melihat hasil query, pastikan MySQL yang dikonfigurasi di Metabase tersedia dan berisi tabel tersebut.

```powershell
cd D:\Dicoding\submission_03\a590_proyek_akhir
docker create --name metabase_submission_03 -p 3000:3000 metabase/metabase:v0.46.4
docker cp .\metabase.db.mv.db metabase_submission_03:/metabase.db/metabase.db.mv.db
docker start metabase_submission_03
```

Setelah container berjalan, buka [http://localhost:3000](http://localhost:3000). Jika nama container `metabase_submission_03` sudah dipakai, hentikan dan hapus container lama terlebih dahulu, lalu jalankan kembali perintah di atas.

## Menjalankan Sistem Machine Learning

Prototype machine learning dibangun menggunakan Streamlit untuk memprediksi probabilitas risiko dropout mahasiswa secara interaktif.

```powershell
cd D:\Dicoding\submission_03
.\.venv\Scripts\Activate.ps1
cd .\a590_proyek_akhir
python -m pip install -r requirements.txt
streamlit run app.py
```

Model siap pakai tersedia pada `model/dropout_pipeline.joblib`. Notebook tidak perlu dijalankan ulang untuk menggunakan prototype yang ada; `notebook.ipynb` digunakan apabila ingin mereproduksi proses analisis dan pelatihan model.

## Deployment Streamlit

[Buka prototype Streamlit](https://seta000002026-09-12144838609261installingbuilddependenciesstar.streamlit.app/)

Prototype dideploy melalui Streamlit Community Cloud dengan entry point `app.py` dan menggunakan artefak model `model/dropout_pipeline.joblib` yang tersedia di repository.

## Conclusion
Berdasarkan hasil analisis data dan pemodelan machine learning pada `notebook.ipynb`:

1. **Proporsi Populasi Mahasiswa** (`notebook.ipynb - section Data Understanding`):
   - Dari 4.424 total mahasiswa, sebanyak 2.209 mahasiswa (49,93%) berhasil lulus (Graduate), 1.421 mahasiswa (32,12%) mengalami Dropout, dan 794 mahasiswa (17,95%) berstatus aktif terdaftar (Enrolled). Tingkat dropout sebesar 32,12% menunjukkan kebutuhan mendesak akan intervensi dini.
2. **Temuan EDA Kunci** (`notebook.ipynb - section Data Understanding`):
   - **Performa Akademik**: Mahasiswa yang mengalami dropout memiliki median nilai mata kuliah semester 2 (`Curricular_units_2nd_sem_grade`) mendekati 0 atau jauh di bawah standar kelulusan, berbeda signifikan dengan kelompok Graduate yang memiliki median di atas 12.
   - **Status Pembayaran SPP**: Mahasiswa yang menunggak pembayaran uang kuliah (`Tuition_fees_up_to_date = 0`) didominasi oleh status Dropout, membuktikan kendala finansial berkorelasi kuat dengan putus studi.
   - **Status Hutang (Debtor)**: Mahasiswa yang berstatus memiliki hutang/tunggakan (`Debtor = 1`) mencatatkan dropout rate sebesar 62,3%, jauh lebih tinggi dibandingkan mahasiswa tanpa hutang (28,1%).
3. **Kinerja Model Terpilih** (`notebook.ipynb - section Modeling & Evaluation`):
   - Model terbaik yang dipilih adalah **Logistic Regression** (dengan preprocessor `ColumnTransformer` dan `class_weight='balanced'`) yang mengungguli RandomForest dan Dummy baseline dalam metrik deteksi dropout.
   - Metrik evaluasi pada data uji (test set):
     - **Recall (Dropout)**: 94,01% (mampu mendeteksi 267 dari 284 mahasiswa dropout pada data uji).
     - **Precision (Dropout)**: 86,69%.
     - **F1-Score (Dropout)**: 90,20%.
     - **ROC-AUC**: 97,57%.
     - **Overall Accuracy**: 92,00%.

### Rekomendasi Action Items
- Prioritaskan outreach kepada mahasiswa dengan performa mata kuliah semester awal yang rendah atau banyak evaluasi yang tidak terselesaikan guna memberikan pendampingan sebelum semester 2 berakhir.
- Lakukan follow-up pembayaran dan konseling bantuan biaya bagi mahasiswa dengan indikator tunggakan atau biaya pendidikan belum mutakhir, mengingat mahasiswa dengan tunggakan memiliki dropout rate melebihi 62%.
- Tawarkan pendampingan akademik terjadwal untuk mahasiswa dengan unit mata kuliah yang disetujui rendah guna meningkatkan completion rate mata kuliah wajib.
- Gunakan prediksi sebagai daftar prioritas review manusia; jangan menjadikannya dasar keputusan akademik otomatis, melainkan sebagai sistem peringatan dini (early warning system) bagi pembimbing akademik.
