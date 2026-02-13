FROM python:3.12-slim

# 2. Gerekli sistem araçlarını kur (psycopg2 ve diğerleri için şart)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Kütüphaneleri yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Tüm dosyaları kopyala
COPY . .

# 5. Render'ın beklediği portu aç
EXPOSE 10000


CMD ["sh", "-c", "python ingest.py && uvicorn main:app --host 0.0.0.0 --port 10000"]
