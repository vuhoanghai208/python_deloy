from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import faiss
import pickle
import numpy as np
import os
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

# 2. Cấu hình OpenAI Client
# Lấy Key từ biến môi trường trên Render
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 3. API Health Check (Sửa lỗi 404 Ping)
@app.get("/")
def read_root():
    return {"status": "OpenAI Server is running"}

# 4. Load Database (Bắt buộc phải là DB tạo bằng OpenAI)
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
        print("⚠️ Lỗi: Không tìm thấy file dữ liệu. Hãy đảm bảo bạn đã chạy build_db_openai.py!")
except Exception as e:
    print(f"❌ Lỗi khi tải DB: {e}")

# 5. Hàm tìm kiếm Vector (Dùng OpenAI Embeddings)
def vector_search(query):
    if not index or not documents:
        return ""

    try:
        # Tạo vector từ câu hỏi của người dùng
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        q_vec = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(q_vec) 
        
        # Tìm 5 đoạn luật khớp nhất
        scores, indices = index.search(q_vec, 5)
        
        relevant_docs = []
        for i, score in enumerate(scores[0]):
            if score >= 0.35: # Ngưỡng lọc độ chính xác
                relevant_docs.append(documents[indices[0][i]])
        
        if relevant_docs:
            return "\n---\n".join(relevant_docs)
        else:
            return ""
            
    except Exception as e:
        print(f"Lỗi tìm kiếm: {e}")
        return ""

# 6. API Xử lý Chat
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(request: ChatRequest):
    user_input = request.prompt
    
    # Bước A: Tìm kiếm dữ liệu luật
    context = vector_search(user_input)
    
    # Bước B: Xây dựng Prompt
    if context:
        system_content = f"""
        Bạn là Trợ lý Pháp luật Giao thông Việt Nam (Nghị định 168/2024).
        Dưới đây là thông tin trích xuất từ văn bản luật:
        ---------------------
        {context}
        ---------------------
        YÊU CẦU:
        1. CHỈ sử dụng thông tin trên để trả lời.
        2. Ghi rõ mức phạt tiền cụ thể (nếu có).
        3. Trả lời ngắn gọn, súc tích.
        """
    else:
        system_content = """
        Bạn là Trợ lý Giao thông. Hiện tại trong cơ sở dữ liệu không có thông tin về câu hỏi này.
        Hãy khéo léo xin lỗi và gợi ý người dùng hỏi về các lỗi vi phạm phổ biến.
        """

    # Bước C: Gọi GPT-4o-mini để trả lời
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Model bạn yêu cầu
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_input}
            ],
            temperature=0.3, # Giữ cho câu trả lời ổn định, ít bịa đặt
            max_tokens=500
        )
        
        return {"answer": response.choices[0].message.content}
        
    except Exception as e:
        print(f"Lỗi OpenAI: {e}")
        return {"answer": "Hệ thống đang bận, vui lòng thử lại sau."}
