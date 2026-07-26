# HAgent — HAgent Integration

## Architecture (default)

```text
Chat UI → Bridge (FastAPI :9900)
            │  JWT + Mongo conversations + WorldState
            ▼
         toolkit app.py (:8585)
            │  POST /api/v1/chat/agent-run
            ▼
         LangGraph multi-agent
            (hierarchy → campaign / tools → synthesizer)
            ▼
         HAutoML jobs / MinIO / Kafka workers
```

| Env | Default | Meaning |
|---|---|---|
| `HAGENT_AGENT_RUN_URL` | `{HAUTOML}/api/v1/chat/agent-run` | Bridge → toolkit agent |
| `HAGENT_CONFIG` | `hagent/hagent.yaml` | Central YAML |
| `OLLAMA_BASE_URL` | host gateway | Local LLM |

## Docker

From `src/backend`:

```bash
# Core stack
docker compose up --build -d toolkit hagent_bridge mongo kafka minio

# With workers
docker compose --profile worker up --build -d
```

Images:

- `hautoml.toolkit.dockerfile` — MỘT image backend dùng chung
  toolkit / worker / nano (command đặt trong compose)
- `hagent/bridge/Dockerfile` — Bridge mỏng (HTTP client tới toolkit)

## Tools

- LangChain tools: `hagent/agent/tools/automl_tools.py` (source of truth)

## Health

- Bridge: `GET http://localhost:5360/api/v1/chat/health`
- Toolkit: `GET http://localhost:5370/home`
- Agent invoke (auth): `POST /api/v1/chat/agent-run`

## Ghi chú công nghệ

Kiến trúc multi-agent tham khảo mẫu điều phối agent hiện đại (LangGraph
harness); toàn bộ mã tích hợp nằm trong `hagent/agent/` của dự án này.
