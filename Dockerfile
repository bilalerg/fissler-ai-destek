FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıklarını yükle (pdf işleme için gerekli olabilir)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt'yi kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tüm proje dosyalarını kopyala
COPY . .

# FAISS indeksi zaten var, ekstra işlem yok

# Render'ın beklediği port
EXPOSE 10000

# FastAPI uygulamasını başlat
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
