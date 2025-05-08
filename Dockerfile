# 1. Python 베이스 이미지 사용
FROM python:3.10-slim

# 2. 작업 디렉토리 생성 및 이동
WORKDIR /app

# 3. requirements.txt 복사 및 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
COPY . .

# 5. FastAPI 실행 (uvicorn)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
