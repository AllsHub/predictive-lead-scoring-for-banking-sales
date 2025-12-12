-----
# Predictive Lead Scoring for Banking Sales

Repositori ini berisi model untuk memprediksi potensi nasabah deposito berjangka (*Term Deposit*). Service ini dibangun menggunakan **FastAPI**, **XGBoost** (Calibrated), dan **Docker**, siap untuk diintegrasikan dengan sistem Backend utama.

---

## Struktur File

| File | Fungsi |
| :--- | :--- |
| `main.py` | **Entry Point API (Server).** Kode utama untuk endpoint `POST /predict`. |
| `transformers.py` | Modul Feature Engineering custom (wajib ada agar model bisa berjalan). |
| `train_model.py` | Skrip untuk melatih model & menghasilkan file `.pkl` (Arsip/Dokumentasi). |
| `model_deposito_siap_pakai.pkl` | **Otak Model.** Model XGBoost yang sudah dilatih dan dikalibrasi. |
| `Dockerfile` | Konfigurasi kontainer untuk deployment (Production Ready). |
| `requirements.txt` | Daftar library Python yang dibutuhkan. |

---

## Cara Deploy (Untuk Tim Backend/DevOps)

Service ini sudah di-*containerize*. Tidak perlu setup environment Python manual.

### 1. Build Docker Image
Jalankan perintah berikut di terminal server:
```bash
docker build -t ai-service .
````

### 2\. Run Container

Jalankan service di port 8000 (atau port lain sesuai konfigurasi gateway):

```bash
docker run -d -p 8000:8000 --name ai-container ai-service
```

### 3\. Cek Status

Akses endpoint health check untuk memastikan service hidup:
`GET http://localhost:8000/`

-----

## Dokumentasi API

### Endpoint Prediksi

  * **URL:** `POST /predict`
  * **Content-Type:** `application/json`

### Contoh Request Body (Payload)

Kirimkan data nasabah mentah (Raw Data) dari form input user.

```json
{
  "age": 35,
  "job": "admin.",
  "marital": "married",
  "education": "university.degree",
  "default": "no",
  "housing": "yes",
  "loan": "no",
  "contact": "cellular",
  "month": "may",
  "day_of_week": "mon",
  "campaign": 1,
  "pdays": 999,
  "previous": 0,
  "poutcome": "nonexistent",
  "emp_var_rate": 1.1,
  "cons_price_idx": 93.994,
  "cons_conf_idx": -36.4,
  "euribor3m": 4.857,
  "nr_employed": 5191.0
}
```

### Contoh Response (Output)

Service akan mengembalikan **Skor Probabilitas** dan **Label Prioritas**.

```json
{
  "prediction": 0,
  "score": 0.35,
  "tier": "TIER_1",
  "label_code": "HIGH_PRIORITY",
  "description": "Probability: 35.00%"
}
```

-----

## Panduan Integrasi (PENTING)

Untuk memaksimalkan fitur **Lead Scoring** di aplikasi Frontend, mohon ikuti panduan berikut:

### 1\. Untuk Backend Engineering

Saat menerima response dari model, mohon simpan dua field ini ke Database utama:

  * **`score` (Float):** Simpan nilai probabilitas murni (0.0 - 1.0). Gunakan kolom ini untuk fitur **Sorting** (Urutkan nasabah dari skor tertinggi ke terendah).
  * **`label_code` (String):** Simpan label kategori (misal: `HIGH_PRIORITY`). Gunakan kolom ini untuk fitur **Filtering** (Tampilkan hanya nasabah prioritas tinggi).

### 2\. Untuk Frontend Engineering (UI/UX)

Saya tidak mengirimkan kode warna (Hex code) dari API agar kalian memiliki kebebasan desain. Mohon petakan `label_code` ke visualisasi berikut (Saran):

| Output API (`label_code`) | Arti Statistik (Data Science) | Rekomendasi Visual |
| :--- | :--- | :--- |
| **`HIGH_PRIORITY`** | Nasabah ini masuk **Top 10%** peluang tertinggi secara historis. | **Warna:** Merah 🔴<br>**Badge:** Hot Lead |
| **`MEDIUM_PRIORITY`** | Nasabah ini masuk **Top 30%** peluang menengah. | **Warna:** Kuning 🟡<br>**Badge:** Warm Lead |
| **`STANDARD_PRIORITY`** | Nasabah dengan peluang standar/rata-rata. | **Warna:** Biru 🔵<br>**Badge:** Cold Lead |

-----

## Catatan Teknis (Data Science)

  * **Model:** XGBoost Classifier dengan Hyperparameter Tuning (Optuna).
  * **Kalibrasi:** Menggunakan `CalibratedClassifierCV` (Sigmoid) untuk memastikan probabilitas yang dihasilkan akurat dan dapat dipercaya (*Reliable*).
  * **Threshold:** Penentuan Tier (High/Medium/Standard) menggunakan metode **Persentil** dari data training historis, bukan angka statis sembarangan. Untuk menjamin label "High Priority" selalu relevan dengan kondisi pasar.

-----
