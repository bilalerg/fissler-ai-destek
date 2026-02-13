import os
import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from pydantic import BaseModel

# .env dosyasını yükle
load_dotenv()

app = FastAPI()

# Veritabanı URL'ini al
DB_URL = os.getenv("SUPABASE_URL")

# Frontend'den gelecek veri modeli (HTML'deki JSON ile birebir aynı olmalı)
class UserData(BaseModel):
    full_name: str
    email: str
    product_model: str

# 1. Ana Sayfa (Giriş Ekranı)
@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        # static klasöründeki index.html'i oku ve gönder
        with open("static/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Hata: static/index.html dosyası bulunamadı!</h1>",
            status_code=404
        )

# 2. Kayıt API'si (Frontend buraya istek atar)
@app.post("/api/register")
async def register_user(data: UserData):
    conn = None
    try:
        # Veritabanına bağlan
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()

        # İsim Soyisim Ayırma (Basit mantık)
        parts = data.full_name.strip().split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""

        # --- ADIM 1: Kullanıcı Kontrolü ---
        cur.execute("SELECT id FROM users WHERE email = %s", (data.email,))
        user_res = cur.fetchone()

        if user_res:
            # Kullanıcı varsa ID'sini al ve adını güncelle (belki adı değişmiştir)
            user_id = user_res[0]
            cur.execute(
                "UPDATE users SET first_name=%s, last_name=%s WHERE id=%s",
                (first_name, last_name, user_id)
            )
        else:
            # Kullanıcı yoksa yeni oluştur
            cur.execute(
                "INSERT INTO users (email, first_name, last_name) VALUES (%s, %s, %s) RETURNING id",
                (data.email, first_name, last_name)
            )
            user_id = cur.fetchone()[0]

        # --- ADIM 2: Ürün Kaydı (Log amaçlı) ---
        # Kullanıcının seçtiği modeli veritabanına ekleyelim
        cur.execute(
            "INSERT INTO user_products (user_id, product_model, created_at) VALUES (%s, %s, NOW())",
            (user_id, data.product_model)
        )

        conn.commit()
        cur.close()

        # Başarılı dönerse HTML tarafı kullanıcıyı /chat sayfasına yönlendirecek
        return {"status": "success", "user_id": str(user_id)}

    except Exception as e:
        print(f"DB Error: {e}")
        if conn:
            conn.rollback()
        # Frontend'e hata mesajını dön
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

    finally:
        if conn:
            conn.close()

# 3. Chainlit Entegrasyonu (En sonda olmalı)
# target="app.py" -> Chainlit kodlarının olduğu dosya adı
mount_chainlit(app=app, target="app.py", path="/chat")