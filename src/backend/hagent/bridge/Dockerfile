FROM python:3.12-slim

LABEL org.opencontainers.image.title="HAgent Bridge"
LABEL org.opencontainers.image.description="JWT/chat store bridge → DeerFlow-AutoML (LangGraph on toolkit)"

WORKDIR /app

# Bridge deps (HTTP to toolkit agent-run; no LangGraph needed in bridge process)
COPY hagent/bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Full hagent package (config, world helpers, bridge app)
COPY hagent/ /app/hagent/
RUN echo "# HAgent package" > /app/hagent/__init__.py

EXPOSE 9900

ENV BRIDGE_HOST=0.0.0.0
ENV BRIDGE_PORT=9900
ENV PYTHONPATH=/app
ENV HAGENT_RUNTIME_MODE=deerflow
# Override in compose: HAGENT_DEERFLOW_URL=http://toolkit:8585/api/v1/chat/agent-run

CMD ["uvicorn", "hagent.bridge.app:hagent_bridge", "--host", "0.0.0.0", "--port", "9900"]
