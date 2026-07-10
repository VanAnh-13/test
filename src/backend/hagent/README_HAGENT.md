# HAgent — DeerFlow-AutoML Integration

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
| `HAGENT_RUNTIME_MODE` | `deerflow` | `deerflow` or `openclaw` |
| `HAGENT_DEERFLOW_URL` | `{HAUTOML}/api/v1/chat/agent-run` | Bridge → toolkit agent |
| `HAGENT_CONFIG` | `hagent/hagent.yaml` | Central YAML |
| `OLLAMA_BASE_URL` | host gateway | Local LLM |

## Docker

From `src/backend`:

```bash
# DeerFlow only (recommended)
docker compose up --build -d toolkit hagent_bridge mongo kafka minio

# With workers
docker compose --profile worker up --build -d

# Legacy OpenClaw
HAGENT_RUNTIME_MODE=openclaw docker compose --profile openclaw --profile worker up --build -d
```

Images:

- `hautoml.toolkit.dockerfile` — API + DeerFlow agent code
- `hagent/bridge/Dockerfile` — Bridge only (HTTP client to toolkit)

## Skills

- DeerFlow tools: `hagent/agent/tools/automl_tools.py`
- OpenClaw skill docs/CLI: `hagent/skills/hautoml/`
  - `SKILL.md`, `tools.yaml`, `scripts/hautoml_tools.py`

## Health

- Bridge: `GET http://localhost:5360/api/v1/chat/health`
- Toolkit: `GET http://localhost:5370/home`
- Agent invoke (auth): `POST /api/v1/chat/agent-run`

## Deprecated

- Toolkit auto-start `hagent/proxy.py` + always-on OpenClaw (now optional profile)
- Bridge-only OpenClaw gateway as default path
