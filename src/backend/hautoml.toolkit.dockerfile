FROM python:3.10.12

LABEL org.opencontainers.image.title="HAutoML Toolkit + DeerFlow-AutoML"
LABEL org.opencontainers.image.description="HAutoML API + LangGraph multi-agent (HAgent)"

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      vim curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Optional: Node/OpenClaw for legacy profile (kept for --profile openclaw hosts)
# Skip heavy node install in default path — OpenClaw runs in separate image.
# Uncomment if you need openclaw CLI inside toolkit:
# RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
#     && apt-get install -y nodejs \
#     && npm install -g openclaw

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV HAGENT_CONFIG=/app/hagent/hagent.yaml
ENV HAGENT_RUNTIME_MODE=deerflow

# HAutoML API (+ /api/v1/chat/* DeerFlow agent)
EXPOSE 8585

# Override in compose if needed
CMD ["python", "app.py"]
