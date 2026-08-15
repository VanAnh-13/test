# ═══════════════════════════════════════════════════════════
# HAutoML backend image — MỘT image dùng chung cho 3 service:
#   toolkit (python -m server.application)  ·  worker (uvicorn cluster.worker:app)
#   nano    (python automl/demo_gradio.py)
# Command đặt trong docker-compose.yaml; image chỉ lo deps + code.
# (worker.dockerfile / hautoml.nano.dockerfile cũ đã bỏ — cùng
#  requirements.txt thì tách image chỉ tạo drift.)
#
# Khác bản gốc (commit 2ae2739):
#   - python:3.12-slim — khớp môi trường 3.12.13 chạy test/benchmark,
#     base nhỏ hơn ~800MB so với python:3.10.12 đầy đủ
#   - BỎ Node.js + gateway CLI legacy — runtime duy nhất là LangGraph
#     in-process, không cần binary ngoài
#   - Layer caching đúng: requirements trước, code sau
#   - apt libs cho wheel trên slim: libgl1+libglib2.0-0 (opencv),
#     libgomp1 (xgboost OpenMP); curl để debug/healthcheck tay
# ═══════════════════════════════════════════════════════════

FROM python:3.12-slim

LABEL org.opencontainers.image.title="HAutoML Backend"
LABEL org.opencontainers.image.description="FastAPI toolkit + sklearn workers + LangGraph HAgent multi-agent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HAGENT_CONFIG=/app/hagent/config/hagent.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps trước — đổi code không phải cài lại toàn bộ pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code sau (.dockerignore loại .env/.venv/tests/paper — KHÔNG bake secret)
COPY . .

# HAutoML API (+ /api/v1/chat/* HAgent agent)
EXPOSE 8585

# Mặc định chạy toolkit; worker/nano ghi đè command trong compose
CMD ["python", "-m", "server.application"]
