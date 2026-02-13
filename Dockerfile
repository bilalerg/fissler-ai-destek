FROM python:3.12

WORKDIR /app

# Dosyaları kopyala
COPY . .

# Kütüphaneleri yükle
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face portu
EXPOSE 7860

# Önce veritabanını oluştur, sonra siteyi aç
CMD python ingest.py && chainlit run app.py --host 0.0.0.0 --port 7860