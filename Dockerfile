# Gunakan base image Python yang ringan
FROM python:3.9-slim

# Set folder kerja di dalam container
WORKDIR /app

# Copy file requirements dulu (agar cache efisien)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file kode dan model ke dalam container
COPY main.py .
COPY model_deposito_siap_pakai.pkl .

# Buka port 8000 (Standar FastAPI)
EXPOSE 8000

# Perintah untuk menjalankan aplikasi saat container nyala
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]