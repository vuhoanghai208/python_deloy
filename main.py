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

# 2. CẤU HÌNH API KEYS (HYBRID)

# A. Key OpenAI (Dùng để TÌM KIẾM - Embedding)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# B. Key Google (Dùng để TRẢ LỜI - Chat Generative)
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
    return {"status": "Hybrid Server is running"}

# 4. Load Database
print("📥 Đang tải cơ sở dữ liệu luật...")
index = None
documents = None

try:
    if os.path.exists("luat_vn.index") and os.path.exists("luat_vn.pkl"):
        index = faiss.read_index("luat_vn.index")
        with open("luat_vn.pkl", "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Đã tải xong! Tổng cộng {len(documents)} đoạn luật.")
    else:
        print("⚠️ Lỗi: Không tìm thấy file dữ liệu.")
except Exception as e:
    print(f"❌ Lỗi khi tải DB: {e}")

# 5. HÀM TÌM KIẾM (OpenAI Embedding)
def vector_search(query):
    if not index or not documents:
        return ""
    try:
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        q_vec = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(q_vec) 
        scores, indices = index.search(q_vec, 5)
        
        relevant_docs = []
        for i, score in enumerate(scores[0]):
            if score >= 0.35: 
                relevant_docs.append(documents[indices[0][i]])
        
        if relevant_docs:
            return "\n---\n".join(relevant_docs)
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
    
    # BƯỚC 1: TÌM KIẾM
    context = vector_search(user_input)
    
    # BƯỚC 2: XÁC ĐỊNH NGUỒN DỮ LIỆU & CẢNH BÁO
    if context:
        source_instruction = f"Sử dụng thông tin sau để trả lời:\n{context}"
        footer_warning = ""
    else:
        source_instruction = "Hiện tại không tìm thấy trong văn bản luật nạp sẵn. Hãy dùng kiến thức chung của bạn về Luật Giao thông Việt Nam (Nghị định 100/2019, 123/2021) để trả lời."
        footer_warning = "\n\n⚠️ _Lưu ý: Thông tin này dựa trên kiến thức tổng hợp, bạn nên tra cứu văn bản gốc để đối chiếu._"

    # BƯỚC 3: TẠO PROMPT (Cấu hình trình bày đẹp)
    system_prompt = f"""
    Bạn là Trợ lý AI Giao thông Việt Nam thân thiện và chuyên nghiệp.

    {source_instruction}

    QUY TẮC TRÌNH BÀY (BẮT BUỘC TUÂN THỦ):
    1. **ĐỊNH DẠNG:**
       - **TUYỆT ĐỐI KHÔNG** dùng dấu sao (*) ở đầu dòng danh sách. Nó gây xấu giao diện.
       - Hãy dùng dấu gạch ngang (-) hoặc số thứ tự (1., 2.) cho các danh sách.
       - Dùng **In đậm** (bọc trong 2 dấu sao) cho: Số tiền phạt, Tên lỗi vi phạm, Các từ khóa quan trọng.
    
    2. **BỐ CỤC & KHOẢNG CÁCH:**
       - Giữa các ý chính phải có **một dòng trống** để tạo độ thoáng.
       - Không viết một đoạn văn quá dài (trên 5 dòng). Hãy ngắt nhỏ ra.

    3. **EMOJI & SINH ĐỘNG:**
       - Luôn thêm Emoji phù hợp (🚗, 🛵, 🛑, 💰, 👮, ⚠️, ✅) vào đầu các ý chính hoặc tiêu đề.
    
    4. **NỘI DUNG:**
       - Đi thẳng vào vấn đề. Không vòng vo.
       - Nếu câu hỏi về xử phạt: **PHẢI** ghi rõ con số cụ thể (Ví dụ: **2.000.000đ - 3.000.000đ**).
    """

    final_prompt = f"Người dùng hỏi: {user_input} {footer_warning}"

    # BƯỚC 4: GỌI GEMINI TRẢ LỜI
    global key_index
    for i in range(len(GOOGLE_KEYS)):
        try:
            current_key = get_current_google_key()
            genai.configure(api_key=current_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            response = model.generate_content(f"{system_prompt}\n\n{final_prompt}")
            return {"answer": response.text}
            
        except Exception as e:
            print(f"⚠️ Lỗi Gemini (Key {i}): {e}")
            key_index += 1
            time.sleep(0.5)
            
    return {"answer": "😔 Hệ thống đang quá tải. Bạn vui lòng thử lại sau giây lát nhé!"}
