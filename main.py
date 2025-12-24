from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import numpy as np
import os
import google.generativeai as genai
from openai import OpenAI
import time

app = FastAPI()

# 1. Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. CẤU HÌNH API KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

GOOGLE_KEYS_STR = os.getenv("GOOGLE_API_KEYS", "")
GOOGLE_KEYS = [k.strip() for k in GOOGLE_KEYS_STR.split(",") if k.strip()]
key_index = 0

def get_current_google_key():
    global key_index
    if not GOOGLE_KEYS: return None
    return GOOGLE_KEYS[key_index % len(GOOGLE_KEYS)]

# 3. Health Check
@app.get("/")
def read_root():
    return {"status": "Hybrid Server (Law + Social) is running"}

# 4. HÀM LOAD DATABASE (Chung cho cả Luật và Xã giao)
def load_db(name_index, name_pkl):
    print(f"📥 Đang tải DB: {name_index}...")
    if os.path.exists(name_index) and os.path.exists(name_pkl):
        try:
            idx = faiss.read_index(name_index)
            with open(name_pkl, "rb") as f:
                docs = pickle.load(f)
            print(f"✅ Đã tải xong {name_index}: {len(docs)} đoạn.")
            return idx, docs
        except Exception as e:
            print(f"❌ Lỗi tải {name_index}: {e}")
            return None, None
    else:
        print(f"⚠️ Không tìm thấy file {name_index} (Bỏ qua).")
        return None, None

# --- LOAD CẢ 2 DB ---
index_luat, docs_luat = load_db("luat_vn.index", "luat_vn.pkl")
index_social, docs_social = load_db("xa_giao.index", "xa_giao.pkl")

# 5. HÀM TÌM KIẾM ĐA NGUỒN (Hybrid Search)
def search_in_index(idx, docs, query_vec, threshold=0.35, top_k=3):
    if not idx or not docs: return []
    scores, indices = idx.search(query_vec, top_k)
    results = []
    for i, score in enumerate(scores[0]):
        if score >= threshold:
            results.append(docs[indices[0][i]])
    return results

def vector_search(query):
    try:
        # Mã hóa câu hỏi (Dùng OpenAI)
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        q_vec = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(q_vec) 
        
        # 1. Tìm trong XÃ GIAO (Ưu tiên cao, ngưỡng chặt chẽ hơn để tránh nhầm)
        # Ngưỡng 0.45 để đảm bảo câu xã giao phải khá khớp mới lấy
        social_results = search_in_index(index_social, docs_social, q_vec, threshold=0.45, top_k=2)
        
        # 2. Tìm trong LUẬT (Ngưỡng 0.35)
        law_results = search_in_index(index_luat, docs_luat, q_vec, threshold=0.35, top_k=5)
        
        # Gộp kết quả
        final_results = social_results + law_results
        
        if final_results:
            return "\n---\n".join(final_results)
        else:
            return ""
            
    except Exception as e:
        print(f"❌ Lỗi tìm kiếm: {e}")
        return ""

# 6. API Xử lý Chat
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(request: ChatRequest):
    user_input = request.prompt
    
    # BƯỚC A: TÌM KIẾM DỮ LIỆU
    context = vector_search(user_input)
    
    # BƯỚC B: CHUẨN BỊ PROMPT
    if context:
        source_instruction = f"DỮ LIỆU TÌM ĐƯỢC TỪ KHO KIẾN THỨC:\n{context}"
        footer_warning = ""
    else:
        source_instruction = "Không tìm thấy trong dữ liệu nạp sẵn. Hãy dùng kiến thức chung của bạn về Luật Giao thông (NĐ 100/2019, 123/2021) để trả lời."
        footer_warning = "\n\n⚠️ _(Thông tin tham khảo từ kiến thức tổng hợp)_"

    system_prompt = f"""
    Bạn là Trợ lý AI Giao thông Việt Nam thông minh, hài hước và am hiểu luật.

    {source_instruction}

    HƯỚNG DẪN XỬ LÝ QUAN TRỌNG:
    1. **PHÂN LOẠI DỮ LIỆU:**
       - Nếu dữ liệu tìm được có nhãn `[XÃ GIAO]`: Hãy trả lời theo giọng điệu thân thiện, hài hước hoặc "cà khịa" nhẹ nhàng như trong dữ liệu mẫu.
       - Nếu dữ liệu là LUẬT: Hãy trả lời nghiêm túc, chính xác, ngắn gọn.
       - Nếu có cả hai: Hãy chào hỏi xã giao trước, sau đó trả lời luật.

    2. **QUY TẮC TRÌNH BÀY (MARKDOWN):**
       - **TUYỆT ĐỐI KHÔNG** dùng dấu sao (*) ở đầu dòng danh sách.
       - Dùng dấu gạch ngang (-) cho danh sách.
       - Dùng **In đậm** (bọc trong 2 dấu sao) cho: Số tiền phạt, Tên lỗi, Từ khóa.
       - Giữa các ý chính phải có **một dòng trống**.
       - Luôn thêm Emoji (🚗, 🛵, 🛑, 💰, 👮, 😂, 👋) để sinh động.

    3. **NỘI DUNG:**
       - Nếu là câu hỏi luật: **PHẢI** ghi rõ mức phạt cụ thể (VD: **2.000.000đ**).
       - Nếu là câu hỏi xã giao/trêu đùa: Hãy đối đáp lại thông minh.
    """

    final_prompt = f"Người dùng nói: {user_input} {footer_warning}"

    # BƯỚC C: GỌI GEMINI (Sửa lại model chuẩn 1.5-flash)
    global key_index
    for i in range(len(GOOGLE_KEYS)):
        try:
            current_key = get_current_google_key()
            genai.configure(api_key=current_key)
            
            # Lưu ý: Google chưa có bản 2.5-flash public, dùng 1.5-flash là ổn định nhất
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(f"{system_prompt}\n\n{final_prompt}")
            return {"answer": response.text}
            
        except Exception as e:
            print(f"⚠️ Lỗi Gemini (Key {i}): {e}")
            key_index += 1
            time.sleep(0.5)
            
    return {"answer": "😔 Hệ thống đang quá tải. Bạn vui lòng thử lại sau giây lát nhé!"}
