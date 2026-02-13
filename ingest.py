import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

# --- AYARLAR ---
DATA_PATH = "belgeler"  # Senin klasörün burası
DB_FAISS_PATH = "faiss_index"


def determine_family(filename):
    """Dosya ismine bakarak ürün ailesini belirler."""
    name = filename.lower()

    if "vitaquick" in name:
        return "vitaquick"
    elif "vitavit" in name:
        return "vitavit"
    elif "adamant" in name:
        return "adamant"
    else:
        # Garanti belgesi veya genel kılavuzlar buraya düşer
        return "genel"


def create_vector_db():
    if not os.path.exists(DATA_PATH):
        print(f"❌ '{DATA_PATH}' klasörü bulunamadı! Lütfen klasör ismini kontrol et.")
        return

    print(f"📂 '{DATA_PATH}' klasöründeki PDF'ler taranıyor...")

    all_documents = []

    # Klasördeki tüm PDF'leri bul
    pdf_files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))

    if not pdf_files:
        print("❌ Klasörde hiç PDF dosyası yok.")
        return

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        family = determine_family(filename)

        print(f"   👉 Okunuyor: {filename} [Etiket: {family}]")

        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            # 📌 YENİ: Her sayfaya metadata (etiket) ekle + kaynak dosya adı
            for doc in docs:
                doc.metadata["family"] = family
                doc.metadata["source"] = filename
                doc.metadata["source_file"] = filename  # Ekstra alan (daha net filtreleme için)

            all_documents.extend(docs)
            print(f"      ✅ {len(docs)} sayfa yüklendi")
        except Exception as e:
            print(f"   ⚠️ HATA: {filename} okunamadı. Sebebi: {e}")

    print(f"\n✅ Toplam {len(all_documents)} sayfa yüklendi. Parçalanıyor...")

    # Chunk ayarlarını geniş tutuyoruz (Daha önceki başarımızdan dolayı)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    texts = text_splitter.split_documents(all_documents)
    print(f"💾 {len(texts)} parça (chunk) oluşturuldu. Veritabanına işleniyor...")

    # 📌 DEBUG: İlk chunk'ın metadata'sını görelim
    if texts:
        print(f"\n🔍 ÖRNEK CHUNK METADATA:")
        print(f"   Family: {texts[0].metadata.get('family')}")
        print(f"   Source: {texts[0].metadata.get('source')}")
        print(f"   Source File: {texts[0].metadata.get('source_file')}")
        print(f"   İçerik Önizleme: {texts[0].page_content[:100]}...\n")

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

    print(f"🎉 FAISS veritabanı başarıyla güncellendi! Ayrım yapıldı (Vitaquick/Vitavit/Adamant).")
    print(f"📊 İstatistikler:")
    print(f"   - Toplam PDF: {len(pdf_files)}")
    print(f"   - Toplam Sayfa: {len(all_documents)}")
    print(f"   - Toplam Chunk: {len(texts)}")

    # Aile bazında sayıları göster
    family_counts = {}
    for doc in all_documents:
        family = doc.metadata.get("family", "bilinmiyor")
        family_counts[family] = family_counts.get(family, 0) + 1

    print(f"\n📁 Aile Bazında Dağılım:")
    for family, count in sorted(family_counts.items()):
        print(f"   - {family.upper()}: {count} sayfa")


if __name__ == "__main__":
    create_vector_db()