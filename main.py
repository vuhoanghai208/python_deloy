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
    return {"status": "Hybrid Server is running (OpenAI Search + Gemini Chat)"}

# 4. Load Database (Lưu ý: Phải là DB được tạo bằng OpenAI text-embedding-3-small)
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
        print("⚠️ Lỗi: Không tìm thấy file dữ liệu. Hãy chạy build_db_openai.py!")
except Exception as e:
    print(f"❌ Lỗi khi tải DB: {e}")

# 5. HÀM TÌM KIẾM (Dùng OpenAI Embedding)
def vector_search(query):
    if not index or not documents:
        print("❌ Lỗi: DB chưa được load.")
        return ""

    try:
        # Gọi OpenAI để mã hóa câu hỏi
        # Lưu ý: Model này phải KHỚP với model lúc bạn chạy build_db.py
        response = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        
        # Lấy vector
        q_vec = np.array([response.data[0].embedding]).astype('float32')
        faiss.normalize_L2(q_vec) 
        
        # Tìm kiếm trong FAISS
        scores, indices = index.search(q_vec, 5)
        
        relevant_docs = []
        print(f"🔍 Kết quả tìm kiếm cho: '{query}'")
        for i, score in enumerate(scores[0]):
            if score >= 0.35: # Ngưỡng lọc
                print(f"   - Đoạn {indices[0][i]} (Score: {score:.4f})")
                relevant_docs.append(documents[indices[0][i]])
        
        if relevant_docs:
            return "\n---\n".join(relevant_docs)
        else:
            print("   -> Không tìm thấy đoạn nào khớp > 0.35")
            return ""
            
    except Exception as e:
        # ĐÂY LÀ CHỖ IN RA LỖI TÌM KIẾM CỦA BẠN
        print(f"❌ LỖI TÌM KIẾM (OpenAI Embedding): {e}")
        return ""

# 6. API Xử lý Chat
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(request: ChatRequest):
    user_input = request.prompt
    
    # --- BƯỚC 1: TÌM KIẾM (Dùng OpenAI) ---
    context = vector_search(user_input)
    
    # --- BƯỚC 2: TẠO PROMPT ---
    if context:
        system_prompt = f"""
        Bạn là Trợ lý Pháp luật Giao thông Việt Nam (Nghị định 168/2024).
        Dưới đây là thông tin trích xuất từ văn bản luật:
        ---------------------
        {context}
        ---------------------
        YÊU CẦU:
        1. CHỈ sử dụng thông tin trên để trả lời.
        2. Nếu có mức phạt tiền, hãy ghi rõ con số.
        3. Trả lời ngắn gọn, súc tích, thân thiện.
        """
        final_prompt = f"Người dùng hỏi: {user_input}"
    else:
        # Nếu không tìm thấy luật, vẫn cho phép Gemini chém gió (nhưng cảnh báo)
        # Hoặc trả lời khéo léo như file soucre bạn gửi
        system_prompt = """
        Bạn là Trợ lý Giao thông.
        Người dùng đang hỏi một câu mà trong dữ liệu luật hiện tại KHÔNG tìm thấy.
        Hãy trả lời dựa trên kiến thức chung của bạn nhưng phải thêm câu cảnh báo: "Thông tin này chỉ mang tính tham khảo do chưa tìm thấy trong văn bản luật được cung cấp."
        """
        final_prompt = f"Người dùng hỏi: {user_input}"

    # --- BƯỚC 3: TRẢ LỜI (Dùng Google Gemini - Để tiết kiệm tiền) ---
    global key_index
    for i in range(len(GOOGLE_KEYS)):
        try:
            current_key = get_current_google_key()
            genai.configure(api_key=current_key)
            
            # Dùng model 1.5-flash (Bản ổn định nhất hiện tại)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content(f"{system_prompt}\n\n{final_prompt}")
            
            # Trả về kết quả JSON chuẩn cho Frontend
            return {"answer": response.text}
            
        except Exception as e:
            print(f"⚠️ Lỗi Gemini (Key {i}): {e}")
            key_index += 1
            time.sleep(0.5)
            
    # Nếu tất cả đều lỗi
    return {"answer": "Hệ thống đang quá tải hoặc gặp sự cố kết nối. Vui lòng thử lại sau."}
