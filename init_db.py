import os
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_URL")


def create_database_if_missing():
    """Yerel kurulumda hedef veritabanı yoksa oluşturur."""
    db_name = urlparse(DB_URL).path.lstrip("/")
    admin_url = DB_URL.rsplit("/", 1)[0] + "/postgres"
    conn = psycopg2.connect(admin_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    if not cur.fetchone():
        cur.execute(f'CREATE DATABASE "{db_name}"')
        print(f"✅ '{db_name}' veritabanı oluşturuldu")
    conn.close()


def create_tables():
    with open("schema.sql", encoding="utf-8") as f:
        schema = f.read()
    conn = psycopg2.connect(DB_URL)
    with conn, conn.cursor() as cur:
        cur.execute(schema)
    conn.close()
    print("✅ Tablolar hazır: users, user_products")


if __name__ == "__main__":
    if "localhost" in DB_URL or "127.0.0.1" in DB_URL:
        create_database_if_missing()
    create_tables()
