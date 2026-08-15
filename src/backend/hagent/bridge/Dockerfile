FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

LABEL org.opencontainers.image.title="HAgent Bridge"
LABEL org.opencontainers.image.description="Cổng xác thực và hội thoại cho HAgent"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/hagent \
    HAGENT_CONFIG=/app/hagent/config/hagent.yaml

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && groupadd --gid 10002 hagent-bridge \
    && useradd --uid 10002 --gid 10002 --create-home --home-dir /home/hagent hagent-bridge \
    && install -d -o 10002 -g 10002 /var/lib/hagent \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY hagent/bridge/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Chỉ đưa các module Bridge thực sự dùng vào image; toolkit và Agent Runtime ở image riêng.
COPY --chown=root:root hagent/__init__.py /app/hagent/
COPY --chown=root:root hagent/bridge/*.py /app/hagent/bridge/
COPY --chown=root:root hagent/bridge/routes/ /app/hagent/bridge/routes/
COPY --chown=root:root hagent/config/ /app/hagent/config/
COPY --chown=root:root hagent/core/ /app/hagent/core/
COPY --chown=root:root hagent/observability/ /app/hagent/observability/
COPY --chown=root:root hagent/run/ /app/hagent/run/
COPY --chown=root:root hagent/world/ /app/hagent/world/
RUN find /app -type d -exec chmod 0555 {} + \
    && find /app -type f -exec chmod 0444 {} +

USER 10002:10002

EXPOSE 9900
STOPSIGNAL SIGTERM

CMD ["uvicorn", "hagent.bridge.app:hagent_bridge", "--host", "0.0.0.0", "--port", "9900"]
