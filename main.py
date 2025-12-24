from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import numpy as np
import os
import google.generativeai as genai
from duckduckgo_search import DDGS
import time

app = FastAPI()

# 1. Cấu hình CORS - Cho phép Web từ Vercel truy cập 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Quản lý 10 API Keys (Lấy từ Environment Variables trên Render) 
GOOGLE_KEYS_STR = os.getenv("GOOGLE_API_KEYS", "")
GOOGLE_KEYS = [k.strip() for k in GOOGLE_KEYS_STR.split(",") if k.strip()]
key_index = 0

def get_current_key():
    global key_index
    if not GOOGLE_KEYS: return None
    return GOOGLE_KEYS[key_index % len(GOOGLE_KEYS)]

# 3. Load Thư viện FAISS (Đã build sẵn) [cite: 437, 438]
try:
    index = faiss.read_index("luat_vn.index")
    with open("luat_vn.pkl", "rb") as f:
        documents = pickle.load(f)
except Exception:
    index, documents = None, None

# 4. Chức năng tìm kiếm đa nguồn (Thư viện + Internet) [cite: 440, 442]
def hybrid_search(query):
    context_parts = []
    # A. Tìm trong thư viện nội bộ (FAISS) [cite: 441]
    if index and documents:
        try:
            genai.configure(api_key=get_current_key())
            res = genai.embed_content(model="models/text-embedding-004", content=query, task_type="retrieval_query")
            q_vec = np.array([res['embedding']]).astype('float32')
            faiss.normalize_L2(q_vec)
            scores, indices = index.search(q_vec, 3)
            # Chỉ lấy kết quả có độ khớp cao [cite: 441, 442]
            local_docs = [documents[i] for i, score in enumerate(scores[0]) if score >= 0.4]
            if local_docs:
                context_parts.append("DỮ LIỆU THƯ VIỆN:\n" + "\n".join(local_docs))
        except: pass

    # B. Tìm kiếm Web miễn phí (DuckDuckGo) [cite: 443]
    try:
        with DDGS() as ddgs:
            results = ddgs.text(f"{query} quy định giao thông Việt Nam 2025", max_results=3, region='vn-vn')
            if results:
                web_text = "\n".join([r['body'] for r in results])
                context_parts.append("THÔNG TIN INTERNET:\n" + web_text)
    except: pass
    
    return "\n\n---\n\n".join(context_parts)

# 5. API Endpoint nhận request từ Web 
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(request: ChatRequest):
    user_input = request.prompt
    context = hybrid_search(user_input)
    
    # SYSTEM PROMPT BAO QUÁT 6 CHỦ ĐỀ CHÍNH [cite: 474, 477]
    system_prompt = """
    Bạn là Chuyên gia Cố vấn Pháp luật Giao thông Việt Nam.
    Hãy sử dụng dữ liệu thư viện và internet để trả lời chuyên sâu về 6 vấn đề:
    1. Tầm quan trọng ATGT.
    2. Biển báo & Vạch kẻ đường.
    3. Phòng tránh nguy hiểm.
    4. Xe đạp & Xe đạp điện.
    5. Kỹ năng lái xe máy an toàn.
    6. Giao thông Đường sắt & Đường thủy.
    
    QUY TẮC TRẢ LỜI:
    - Luôn trích dẫn mức phạt tiền và điểm trừ GPLX theo Nghị định 168/2024[cite: 421, 478].
    - Trình bày nghiêm túc, chuyên nghiệp bằng Bullet points hoặc Bảng so sánh[cite: 477, 483, 492].
    - Nếu không tìm thấy trong dữ liệu, hãy dựa trên kiến thức hệ thống chuẩn[cite: 452].
    """

    global key_index
    for _ in range(len(GOOGLE_KEYS)):
        try:
            genai.configure(api_key=get_current_key())
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(f"{system_prompt}\n\nBỐI CẢNH: {context}\n\nCÂU HỎI: {user_input}")
            return {"answer": response.text, "status": "success"}
        except:
            key_index += 1
            time.sleep(0.3)
            
    return {"answer": "Hệ thống đang bảo trì, vui lòng thử lại sau.", "status": "error"}