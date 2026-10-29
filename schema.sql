-- main.py ve app.py'nin kullandığı iki tablo
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    first_name  TEXT,
    last_name   TEXT
);

CREATE TABLE IF NOT EXISTS user_products (
    id                     SERIAL PRIMARY KEY,
    user_id                UUID NOT NULL REFERENCES users(id),
    product_model          TEXT NOT NULL,
    purchase_date          DATE,
    next_maintenance_date  DATE,
    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
