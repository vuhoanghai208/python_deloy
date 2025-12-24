from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import numpy as np
import os
import google.generativeai as genai
import time

app = FastAPI()

# 1. Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Quản lý API Keys
GOOGLE_KEYS_STR = os.getenv("GOOGLE_API_KEYS", "")
GOOGLE_KEYS = [k.strip() for k in GOOGLE_KEYS_STR.split(",") if k.strip()]
key_index = 0

def get_current_key():
    global key_index
    if not GOOGLE_KEYS: return None
    return GOOGLE_KEYS[key_index % len(GOOGLE_KEYS)]

# --- QUAN TRỌNG: THÊM ROUTE NÀY ĐỂ SỬA LỖI 404 KHI PING ---
@app.get("/")
def read_root():
    return {"status": "Server is running", "message": "Hello from Render!"}
# ---------------------------------------------------------

# 3. Load Database Vector
print("📥 Đang tải cơ sở dữ liệu luật (Local)...")
index = None
documents = None

try:
    if os.path.exists("luat_vn.index") and os.path.exists("luat_vn.pkl"):
        index = faiss.read_index("luat_vn.index")
        with open("luat_vn.pkl", "rb") as f:
            documents = pickle.load(f)
        print(f"✅ Đã tải xong! Tổng cộng {len(documents)} đoạn luật.")
    else:
        print("⚠️ Lỗi: Không tìm thấy file dữ liệu. Hãy chạy build_db.py trước!")
except Exception as e:
    print(f"❌ Lỗi khi tải DB: {e}")

# 4. Hàm chỉ tìm kiếm Vector
def vector_search_only(query):
    if not index or not documents:
        return "Hệ thống chưa có dữ liệu luật. Vui lòng liên hệ quản trị viên nạp dữ liệu."

    try:
        genai.configure(api_key=get_current_key())
        
        # MODEL NÀY CỦA GOOGLE, KHÔNG PHẢI OPENAI - CODE ĐÚNG RỒI
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        q_vec = np.array([res['embedding']]).astype('float32')
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
        print(f"Lỗi tìm kiếm Vector: {e}")
        return ""

# 5. API Xử lý Chat
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(request: ChatRequest):
    user_input = request.prompt
    context = vector_search_only(user_input)
    
    if context:
        system_prompt = f"""
        Bạn là Trợ lý Pháp luật Giao thông Việt Nam (Nghị định 168/2024).
        Dưới đây là thông tin trích xuất từ văn bản luật chính xác:
        ---------------------
        {context}
        ---------------------
        YÊU CẦU:
        1. CHỈ sử dụng thông tin được cung cấp ở trên để trả lời.
        2. Nếu thông tin có đề cập mức phạt tiền, hãy ghi rõ con số cụ thể.
        3. Trả lời ngắn gọn, đi thẳng vào vấn đề.
        """
        final_prompt = f"Người dùng hỏi: {user_input}"
    else:
        system_prompt = """
        Bạn là Trợ lý Giao thông.
        Người dùng đang hỏi một câu mà hệ thống dữ liệu luật hiện tại KHÔNG tìm thấy thông tin khớp.
        Hãy trả lời khéo léo rằng: "Xin lỗi, hiện tại trong cơ sở dữ liệu của tôi chưa có thông tin cụ thể về vấn đề này. Bạn có thể hỏi rõ hơn về các lỗi vi phạm phổ biến không?"
        """
        final_prompt = f"Câu hỏi: {user_input}"

    global key_index
    for _ in range(len(GOOGLE_KEYS)):
        try:
            genai.configure(api_key=get_current_key())
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                f"{system_prompt}\n\n{final_prompt}"
            )
            return {"answer": response.text}
        except:
            key_index += 1
            time.sleep(0.5)
            
    return {"answer": "Hệ thống đang bận, vui lòng thử lại sau giây lát."}
