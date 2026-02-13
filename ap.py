import os
import chainlit as cl
import psycopg2
from typing import Annotated, TypedDict, Literal
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs

# LangChain & LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

# ================== AYARLAR ==================
DB_KLASORU = "faiss_index"
MODEL_ADI = "gpt-4o-mini"
DATABASE_URL = os.getenv("SUPABASE_URL")


# ================== VERİTABANI İŞLEMLERİ ==================
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def get_user_info(user_id: str):
    """Veritabanından kullanıcı adını ve ürün modelini çeker."""
    if not user_id:
        return "Misafir", "Bilinmiyor"
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT u.first_name, u.last_name, p.product_model 
            FROM users u
            LEFT JOIN user_products p ON u.id = p.user_id
            WHERE u.id = %s
            ORDER BY p.created_at DESC NULLS LAST
            LIMIT 1;
        """, (user_id,))
        res = cur.fetchone()
        cur.close()
        conn.close()

        if res:
            first = res[0] or ""
            last = res[1] or ""
            full_name = f"{first} {last}".strip()
            model = res[2] or "Bilinmiyor"
            return full_name, model
    except Exception as e:
        print("DB ERROR:", e)
    return "Misafir", "Bilinmiyor"


# ================== TOOLS (ARAÇLAR) ==================

@tool
def search_technical_manual(query: str) -> str:
    """Teknik kılavuzlarda arama yapar. Model ayrımı yaparak sadece ilgili belgeleri ve garanti belgesini getirir."""
    try:
        if not os.path.exists(DB_KLASORU):
            return "HATA: Teknik veritabanı bulunamadı."

        # 1. Session'dan kullanıcının ürün ailesini çek
        state_meta = cl.user_session.get("state_metadata", {})
        user_family = state_meta.get("product_family", "genel")

        # Sorgu güçlendirme
        enhanced_query = f"{user_family} {query}"

        print(f"\n🔍 ARAMA BAŞLADI | Aile: {user_family} | Güçlendirilmiş Sorgu: '{enhanced_query}'")

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(DB_KLASORU, embeddings, allow_dangerous_deserialization=True)

        results = []

        # 2. ARAMA AŞAMASI 1: Kullanıcının modeline özel belgeler
        if user_family != "genel":
            docs_specific = vectorstore.similarity_search(
                enhanced_query,
                k=25,  # Geniş ağ at
                filter={"family": user_family}
            )

            results.extend(docs_specific)
            print(f"   📄 {user_family} belgelerinde {len(docs_specific)} sonuç bulundu.")

            # DEBUG: İlk 3 sonucun metadata'sını göster
            if docs_specific:
                print(f"\n   📊 İLK 3 SONUÇ DETAYI:")
                for i, doc in enumerate(docs_specific[:3]):
                    print(f"   {i + 1}. Family: {doc.metadata.get('family')} | Source: {doc.metadata.get('source')}")
                    print(f"      İçerik: {doc.page_content[:80]}...")

        # 3. ARAMA AŞAMASI 2: Genel belgeler (garanti vb.)
        docs_general = vectorstore.similarity_search(
            query,
            k=3,
            filter={"family": "genel"}
        )
        results.extend(docs_general)
        print(f"   📄 Genel belgelerde {len(docs_general)} sonuç bulundu.")

        if not results:
            print("❌ DEBUG: Hiç sonuç dönmedi.")
            return "Veritabanında bu konuyla ilgili bilgi bulunamadı."

        # Sonuç sayısını sınırla
        if len(results) > 15:
            print(f"   ✂️ Sonuçlar {len(results)} → 15'e düşürüldü")
            results = results[:15]

        # DEBUG bilgileri
        if results:
            print(f"\n📝 FINAL SONUÇ:")
            print(f"   Toplam chunk: {len(results)}")
            print(f"   İlk chunk family: {results[0].metadata.get('family')}")
            print(f"   İlk chunk source: {results[0].metadata.get('source')}")
            print(f"   Önizleme: {results[0].page_content[:100]}...")

        return "\n\n".join([d.page_content for d in results])

    except Exception as e:
        print(f"❌ ARAMA HATASI: {e}")
        import traceback
        traceback.print_exc()
        return f"Arşivde arama yapılamadı: {str(e)}"


@tool
def register_product_tool(product_model: str, purchase_date: str) -> str:
    """
    Ürün kaydı yapar. Tarih formatı YYYY-MM-DD olmalıdır.
    Kullanıcı ID'si sistemden otomatik alınır.
    """
    state_cl = cl.user_session.get("state_metadata")
    user_id = state_cl.get("user_id")

    if not user_id:
        return "❌ Hata: Kullanıcı kimliği bulunamadı. Kayıt yapılamıyor."

    try:
        clean_date = purchase_date.strip()[:10]
        p_date = datetime.strptime(clean_date, "%Y-%m-%d")
        m_date = p_date + timedelta(days=730)  # 2 Yıl garanti

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO user_products (user_id, product_model, purchase_date, next_maintenance_date) VALUES (%s,%s,%s,%s)",
            (user_id, product_model, p_date, m_date)
        )
        conn.commit()
        cur.close()
        conn.close()

        return f"✅ Kayıt Başarılı! Bakım tarihiniz: {m_date.strftime('%d.%m.%Y')} olarak ayarlandı."
    except Exception as e:
        return f"❌ Veritabanı hatası: {str(e)}"


tools = [search_technical_manual, register_product_tool]


# ================== LANGGRAPH KURULUMU ==================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    user_name: str
    user_model: str


def agent_node(state: AgentState):
    llm = ChatOpenAI(model=MODEL_ADI, temperature=0, streaming=True).bind_tools(tools)

    state_meta = cl.user_session.get("state_metadata", {})
    user_family = state_meta.get("product_family", "genel").upper()

    sys_msg = SystemMessage(
        content=f"""Sen Fissler yetkili servis asistanı Ahmet Ustasın.

Müşteri Adı: {state.get('user_name', 'Misafir')}
Müşteri Ürünü: {state.get('user_model', 'Bilinmiyor')}
Ürün Ailesi: {user_family}

🔴 KRİTİK KURAL - MUTLAKA OKU VE UYGULA:

1. **TEKNİK SORULARDA MUTLAKA ARAMA YAP:**
   - Ürün özellikleri (renkli halkalar, kademeler, dolum oranları)
   - Kullanım talimatları (nasıl kullanılır, nasıl temizlenir)
   - Sorun giderme (buhar sızıyor, açılmıyor vb.)
   - Yedek parça bilgileri (conta, valf vb.)

   → Bu tür HERHANGI bir soru geldiğinde MUTLAKA `search_technical_manual` aracını kullan!
   → "Dokümanlarımda bulamadım" DEMEDEN ÖNCE mutlaka ara!

2. **ARAMADAN CEVAPLAYABİLECEKLERİN:**
   Sadece şu sabit bilgiler:
   - Müşteri Hizmetleri Telefon: 444 75 58
   - Adres: Türkali Mh. Ihlamurdere Caddesi 85, 34357 Beşiktaş/İstanbul
   - Web Sitesi: www.fisslermagaza.com.tr
   - Garanti Süresi: 2 yıl (malzeme ve işçilik hataları)
   - Genel güvenlik kuralları (fırın yasak, basınçlı kızartma yasak vb.)

3. **ÖNCELIK KURALI:**
   Arama sonuçlarında {user_family} ürününe ait bilgiler varsa, MUTLAKA ONLARI KULLAN.
   Başka ürün ailelerinden bilgi gelirse GÖRMEZDEN GEL.

4. **DİĞER KURALLAR:**
   - "Lastik" = "Düdüklü Tencere Contası"
   - Fırın: ASLA
   - Basınçlı kızartma: ASLA
   - Doluluk: Bakliyat 1/3, Pirinç 1/2, Normal 2/3
   - Su ile soğutma: Sadece YANDAN
   - Sterilizasyon: KULLANILAMAZ

**ÖNEMLİ:** Bir şeyden emin değilsen veya detay bilgi gerekiyorsa, ÖNCE ara, SONRA cevapla!
"""
    )

    response = llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"


workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app_graph = workflow.compile()


# ================== CHAINLIT ARAYÜZÜ ==================

@cl.on_chat_start
async def start():
    """Sohbet başladığında çalışır."""
    user_id = None

    # URL veya Referer'dan User ID çekme
    user_id = cl.user_session.get("query_params", {}).get("user_id")
    if not user_id:
        try:
            environ = cl.context.session.environ
            referer = environ.get("HTTP_REFERER")
            if referer:
                parsed = urlparse(referer)
                params = parse_qs(parsed.query)
                if "user_id" in params:
                    user_id = params["user_id"][0]
        except Exception as e:
            print(f"Header Parse Hatası: {e}")

    print(f"✅ DEBUG FINAL USER_ID: {user_id}")

    if user_id:
        full_name, model = get_user_info(user_id)
    else:
        full_name, model = "Misafir", "Bilinmiyor"

    # Model ailesini belirleme
    model_lower = model.lower()
    product_family = "genel"

    if "vitaquick" in model_lower:
        product_family = "vitaquick"
    elif "vitavit" in model_lower:
        product_family = "vitavit"
    elif "adamant" in model_lower:
        product_family = "adamant"

    print(f"🎯 Kullanıcı Modeli: {model} -> Tespit Edilen Aile: {product_family}")

    await cl.Message(
        content=f"👋 Merhaba **{full_name}**! **{model}** model tencereniz için teknik asistanı hazır."
    ).send()

    initial_state = {
        "messages": [],
        "user_name": full_name,
        "user_model": model
    }

    cl.user_session.set("graph_state", initial_state)
    cl.user_session.set("state_metadata", {"user_id": user_id, "product_family": product_family})


@cl.on_message
async def main(message: cl.Message):
    current_state = cl.user_session.get("graph_state")
    inputs = {"messages": [HumanMessage(content=message.content)]}
    merged_input = {**current_state, **inputs}

    msg = cl.Message(content="")
    await msg.send()

    try:
        res = await app_graph.ainvoke(merged_input)
        bot_response_message = res["messages"][-1]
        msg.content = bot_response_message.content
        await msg.update()
        cl.user_session.set("graph_state", res)

    except Exception as e:
        print(f"HATA OLUŞTU: {e}")
        msg.content = f"⚠️ Bir hata oluştu: {str(e)}"
        await msg.update()