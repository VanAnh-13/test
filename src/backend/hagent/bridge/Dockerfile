FROM python:3.12-slim

LABEL org.opencontainers.image.title="HAgent Bridge"
LABEL org.opencontainers.image.description="JWT/chat store bridge → HAgent runtime (LangGraph on toolkit)"

WORKDIR /app

# Bridge deps (HTTP to toolkit agent-run; no LangGraph needed in bridge process)
COPY hagent/bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Full hagent package (config, world helpers, bridge app) —
# __init__.py thật đã nằm trong repo, không cần hack echo như bản cũ
COPY hagent/ /app/hagent/

EXPOSE 9900

ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=9900
ENV PYTHONPATH=/app
# Override in compose: HAGENT_AGENT_RUN_URL=http://toolkit:8585/api/v1/chat/agent-run

CMD ["uvicorn", "hagent.bridge.app:hagent_bridge", "--host", "0.0.0.0", "--port", "9900"]
