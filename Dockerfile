FROM python:3.11-slim

RUN groupadd -r nbllm && useradd -r -g nbllm nbllm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

RUN mkdir -p data/chroma_db backend/storage/pdfs && \
    chown -R nbllm:nbllm /app

USER nbllm

EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
