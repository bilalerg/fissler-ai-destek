from langchain_community.embeddings import FastEmbedEmbeddings

# Yerelde çalışan, Türkçe dahil çok dilli embedding modeli (~220 MB, ilk çalıştırmada iner).
# ingest.py ve app.py aynı modeli kullanmak zorunda, yoksa FAISS araması anlamsızlaşır.
EMBEDDING_MODELI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODELI)
