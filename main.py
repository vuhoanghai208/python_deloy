from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import numpy as np
import os
import time
import asyncio
import google.generativeai as genai
from duckduckgo_search import AsyncDDGS  # Tìm kiếm Web bất đồng bộ
from openai import AsyncOpenAI           # OpenAI bất đồng bộ

# ================= 1. CẤU HÌNH APP & KHÓA API =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lấy API Keys từ biến môi trường ---
# 1. Google Gemini Keys (Danh sách nhiều key cách nhau dấu phẩy)
GOOGLE_KEYS_STR = os.getenv("GOOGLE_API_KEYS", "")
GOOGLE_KEYS = [k.strip() for k in GOOGLE_KEYS_STR.split(",") if k.strip()]

# 2. OpenAI Key (Dùng để Embed và làm Fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ CẢNH BÁO: Thiếu OPENAI_API_KEY. Chức năng Search và Fallback sẽ lỗi.")

# Client OpenAI Async
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ================= 2. RATE LIMIT (CHỐNG SPAM) =================
RATE_LIMIT = {}
LIMIT = 10       # Cho phép 10 requests
WINDOW = 60      # Trong 60 giây (1 phút)

def check_rate_limit(ip):
    now = time.time()
    # Dọn dẹp IP cũ
    if ip in RATE_LIMIT:
        RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if now - t < WINDOW]
        if not RATE_LIMIT[ip]:
            del RATE_LIMIT[ip]
            
    RATE_LIMIT.setdefault(ip, [])
    if len(RATE_LIMIT.get(ip, [])) >= LIMIT:
        return False
    RATE_LIMIT[ip].append(now)
    return True

# ================= 3. LOAD CƠ SỞ DỮ LIỆU (LUẬT + XÃ GIAO) =================
def load_faiss_db(index_file, pkl_file):
    try:
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            index = faiss.read_index(index_file)
            with open(pkl_file, "rb") as f:
                docs = pickle.load(f)
            print(f"✅ Đã tải DB: {index_file} ({len(docs)} docs)")
            return index, docs
        else:
            print(f"⚠️ Không tìm thấy file: {index_file}")
            return None, None
    except Exception as e:
        print(f"❌ Lỗi tải DB {index_file}: {e}")
        return None, None

# Load cả 2 DB
index_luat, docs_luat = load_faiss_db("luat_vn.index", "luat_vn.pkl")
index_social, docs_social = load_faiss_db("xa_giao.index", "xa_giao.pkl")

# ================= 4. CÁC HÀM XỬ LÝ TÌM KIẾM (CORE LOGIC) =================

# Hàm tạo Vector từ câu hỏi (Dùng OpenAI text-embedding-3-small)
async def get_embedding_async(text):
    if not openai_client: return None
    try:
        resp = await openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        # Chuyển thành numpy array float32 cho FAISS
        vec = np.array([resp.data[0].embedding]).astype('float32')
        faiss.normalize_L2(vec) # Chuẩn hóa vector
        return vec
    except Exception as e:
        print(f"❌ Lỗi Embedding: {e}")
        return None

# Hàm tìm kiếm trong index cụ thể
def search_index(index, docs, vector, top_k=3, threshold=0.0):
    if not index or not docs or vector is None:
        return []
    
    # Search
    scores, indices = index.search(vector, top_k)
    results = []
    
    # Lọc kết quả theo ngưỡng (threshold)
    for i, score in enumerate(scores[0]):
        if score >= threshold:
            idx = indices[0][i]
            if 0 <= idx < len(docs):
                results.append(docs[idx])
    return results

# Hàm Tìm kiếm Hỗn hợp (Luật + Xã giao + Web)
async def hybrid_search(query):
    context_parts = []
    
    # 1. Tạo vector cho câu hỏi
    q_vec = await get_embedding_async(query)

    # 2. Tìm trong DB XÃ GIAO (Ngưỡng cao để tránh nhầm)
    # Nếu câu hỏi khớp > 45% với câu xã giao thì lấy
    social_res = search_index(index_social, docs_social, q_vec, top_k=2, threshold=0.45)
    if social_res:
        context_parts.append("[KỊCH BẢN XÃ GIAO/GIAO TIẾP]:\n" + "\n".join(social_res))

    # 3. Tìm trong DB LUẬT (Ngưỡng vừa phải)
    law_res = search_index(index_luat, docs_luat, q_vec, top_k=5, threshold=0.35)
    if law_res:
        context_parts.append("[DỮ LIỆU LUẬT & NGHỊ ĐỊNH]:\n" + "\n".join(law_res))

    # 4. Tìm kiếm Internet (DuckDuckGo) - Chỉ chạy khi không tìm thấy luật trong DB
    # Hoặc luôn chạy để bổ sung tin tức mới (tùy chọn)
    if not law_res: 
        try:
            ddg_res = await AsyncDDGS().text(
                f"{query} luật giao thông Việt Nam 2025",
                max_results=2,
                region="vn-vn"
            )
            if ddg_res:
                web_text = "\n".join([r['body'] for r in ddg_res])
                context_parts.append("[THÔNG TIN INTERNET (THAM KHẢO)]:\n" + web_text)
        except Exception:
            pass # Lỗi web thì bỏ qua

    return "\n\n---\n\n".join(context_parts)

# ================= 5. GỌI AI (GEMINI -> FALLBACK GPT) =================

# Gọi Google Gemini (Async)
async def call_gemini(api_key, prompt):
    genai.configure(api_key=api_key)
    # Dùng 1.5-flash cho nhanh và ổn định
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = await model.generate_content_async(prompt)
    return response.text

# Gọi OpenAI GPT (Async) - Dùng làm Fallback
async def call_gpt_fallback(prompt):
    if not openai_client:
        raise RuntimeError("Không có OpenAI Key để chạy Fallback")
    
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini", # Rẻ và nhanh
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia Luật Giao thông VN."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# ================= 6. API ENDPOINT =================
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(req: Request, body: ChatRequest):
    # 1. Check Rate Limit
    ip = req.client.host
    if not check_rate_limit(ip):
        raise HTTPException(429, "Bạn gửi quá nhiều yêu cầu. Vui lòng đợi 1 phút.")

    user_input = body.prompt.strip()
    if not user_input:
        return {"answer": "Bạn chưa nhập câu hỏi nào cả! 😅"}

    # 2. Tìm kiếm dữ liệu (Search)
    context = await hybrid_search(user_input)

    # 3. Tạo Prompt
    source_warning = ""
    if not context:
        source_warning = "\n⚠️ *Lưu ý: Không tìm thấy dữ liệu trong thư viện. Câu trả lời dựa trên kiến thức tổng hợp của AI.*"
        
    system_instruction = """
    VAI TRÒ: Bạn là Trợ lý AI Cố vấn Pháp luật Giao thông Việt Nam & Bạn đường tin cậy.
    
    NHIỆM VỤ:
    1. Nếu là câu hỏi XÃ GIAO (Chào hỏi, trêu đùa, hỏi tên...):
       - Trả lời thân thiện, hài hước, ngắn gọn.
       
    2. Nếu là câu hỏi LUẬT/KIẾN THỨC:
       - Dựa tuyệt đối vào [NGỮ CẢNH THAM KHẢO] bên dưới.
       - Trích dẫn Nghị định 100/2019 hoặc 123/2021 hoặc 168/2024.
       - Nêu rõ: Mức phạt tiền (In đậm) và Hình phạt bổ sung (Tước bằng, giam xe...).
       - Trình bày dạng danh sách (Bullet points) dễ đọc.
    
    3. NGUYÊN TẮC:
       - Không bịa đặt mức phạt.
       - Luôn dùng Emoji (🚗, 👮, 💰) để sinh động.
    """

    final_prompt = f"""
    [SYSTEM]
    {system_instruction}

    [NGỮ CẢNH THAM KHẢO TỪ DATABASE & INTERNET]
    {context if context else "Không có dữ liệu cụ thể."}

    [CÂU HỎI NGƯỜI DÙNG]
    {user_input}

    [TRẢ LỜI]
    """

    # 4. CHIẾN THUẬT GỌI AI: GEMINI XOAY VÒNG -> GPT FALLBACK
    
    # --- GIAI ĐOẠN 1: Thử tất cả key Gemini ---
    for idx, key in enumerate(GOOGLE_KEYS):
        try:
            answer = await call_gemini(key, final_prompt)
            return {
                "answer": answer + source_warning,
                "model": "gemini",
                "key_used": idx, # Để debug xem đang dùng key nào
                "status": "success"
            }
        except Exception as e:
            print(f"⚠️ Gemini Key {idx} lỗi: {e}. Đang thử key tiếp theo...")
            continue # Thử key kế tiếp

    # --- GIAI ĐOẠN 2: Nếu tất cả Key Gemini đều lỗi -> Dùng GPT ---
    print("🚨 TẤT CẢ KEY GEMINI ĐỀU LỖI. CHUYỂN SANG GPT FALLBACK!")
    try:
        answer = await call_gpt_fallback(final_prompt)
        return {
            "answer": answer + source_warning,
            "model": "gpt-fallback",
            "status": "success"
        }
    except Exception as e:
        print(f"❌ GPT Fallback cũng lỗi: {e}")
        return {
            "answer": "Hệ thống đang quá tải và bảo trì. Bạn vui lòng thử lại sau 1 phút nhé! 😔",
            "status": "error"
        }

# ================= 7. HEALTH CHECK =================
@app.get("/")
def health_check():
    return {
        "status": "online",
        "mode": "Hybrid (Law + Social)",
        "gemini_keys": len(GOOGLE_KEYS),
        "gpt_ready": bool(OPENAI_API_KEY),
        "db_law": index_luat is not None,
        "db_social": index_social is not None
    }
