# 1. 파이썬 3.11 버전을 기반으로 서버 환경을 시작합니다.
FROM python:3.11-slim

# 2. 컨테이너 내부에 작업 공간(/app)을 만듭니다.
WORKDIR /app

# 3. 의존성 먼저 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 애플리케이션 복사
COPY . /app

# 5. 포트 오픈
EXPOSE 8000

#6. 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
