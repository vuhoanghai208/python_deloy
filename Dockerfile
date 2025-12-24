# 1. Sử dụng hình ảnh Python nhẹ để tiết kiệm tài nguyên Render
FROM python:3.10-slim

# 2. Thiết lập thư mục làm việc
WORKDIR /app

# 3. Cài đặt thư viện hệ thống cần thiết cho FAISS (libgomp1)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy và cài đặt các thư viện Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy toàn bộ mã nguồn và dữ liệu (luat_vn.index, luat_vn.pkl)
COPY . .

# 6. Chạy ứng dụng FastAPI bằng uvicorn trên cổng Render cung cấp
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]