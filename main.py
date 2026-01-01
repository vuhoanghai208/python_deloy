from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import faiss
import pickle
import numpy as np
import os
import time
import asyncio

from google import genai
from ddgs import DDGS
from openai import AsyncOpenAI

# ================= 1. APP CONFIG =================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. API KEYS =================
GOOGLE_KEYS_STR = os.getenv("GOOGLE_API_KEYS", "")
GOOGLE_KEYS = [k.strip() for k in GOOGLE_KEYS_STR.split(",") if k.strip()]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ================= 3. RATE LIMIT =================
RATE_LIMIT = {}
LIMIT = 50
WINDOW = 60

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    RATE_LIMIT.setdefault(ip, [])
    RATE_LIMIT[ip] = [t for t in RATE_LIMIT[ip] if now - t < WINDOW]

    if len(RATE_LIMIT[ip]) >= LIMIT:
        return False

    RATE_LIMIT[ip].append(now)
    return True

# ================= 4. LOAD FAISS =================
def load_faiss_db(index_file, pkl_file):
    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        return None, None
    index = faiss.read_index(index_file)
    with open(pkl_file, "rb") as f:
        docs = pickle.load(f)
    return index, docs

index_luat, docs_luat = load_faiss_db("luat_vn.index", "luat_vn.pkl")
index_social, docs_social = load_faiss_db("xa_giao.index", "xa_giao.pkl")

# ================= 5. EMBEDDING =================
async def get_embedding_async(text: str):
    if not openai_client:
        return None
    resp = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec

def search_index(index, docs, vector, top_k=3, threshold=0.0):
    if index is None or vector is None:
        return []
    scores, ids = index.search(vector, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if score >= threshold and 0 <= idx < len(docs):
            results.append(docs[idx])
    return results

# ================= 6. DUCKDUCKGO (SAFE) =================
def ddg_search_sync(query: str):
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=2, region="vn-vn"))

async def ddg_search(query: str):
    return await asyncio.to_thread(ddg_search_sync, query)

# ================= 7. HYBRID SEARCH =================
async def hybrid_search(query: str) -> str:
    parts = []
    vec = await get_embedding_async(query)

    social = search_index(index_social, docs_social, vec, 2, 0.45)
    if social:
        parts.append("[XÃ GIAO]\n" + "\n".join(social))

    law = search_index(index_luat, docs_luat, vec, 5, 0.35)
    if law:
        parts.append("[LUẬT]\n" + "\n".join(law))
    else:
        try:
            web = await ddg_search(f"{query} luật giao thông Việt Nam 2025")
            if web:
                parts.append("[WEB]\n" + "\n".join(r["body"] for r in web if "body" in r))
        except Exception:
            pass

    return "\n\n---\n\n".join(parts)

# ================= 8. GEMINI CLIENT POOL =================
GEMINI_CLIENTS = {}

def get_gemini_client(api_key: str):
    if api_key not in GEMINI_CLIENTS:
        GEMINI_CLIENTS[api_key] = genai.Client(api_key=api_key)
    return GEMINI_CLIENTS[api_key]

async def call_gemini(api_key: str, prompt: str) -> str:
    client = get_gemini_client(api_key)
    resp = await client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return resp.text

# ================= 9. GPT FALLBACK =================
async def call_gpt_fallback(prompt: str) -> str:
    if not openai_client:
        raise RuntimeError("Missing OpenAI API key")
    resp = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là chuyên gia Luật Giao thông Việt Nam."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return resp.choices[0].message.content

# ================= 10. API =================
class ChatRequest(BaseModel):
    prompt: str

@app.post("/api/process")
async def process_data(req: Request, body: ChatRequest):
    ip = req.client.host
    if not check_rate_limit(ip):
        raise HTTPException(429, "Rate limit exceeded")

    user_input = body.prompt.strip()
    if not user_input:
        return {"answer": "Bạn chưa nhập câu hỏi."}

    context = await hybrid_search(user_input)

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

[CONTEXT]
{context or "Không có dữ liệu tham khảo."}

[QUESTION]
{user_input}

[ANSWER]
"""

    for idx, key in enumerate(GOOGLE_KEYS):
        try:
            answer = await call_gemini(key, final_prompt)
            return {
                "answer": answer,
                "model": "gemini",
                "key_used": idx
            }
        except Exception:
            continue

    answer = await call_gpt_fallback(final_prompt)
    return {
        "answer": answer,
        "model": "gpt-fallback"
    }

# ================= 11. HEALTH =================
@app.get("/")
def health():
    return {
        "status": "ok",
        "gemini_keys": len(GOOGLE_KEYS),
        "openai": bool(OPENAI_API_KEY),
        "db_law": index_luat is not None,
        "db_social": index_social is not None
    }
