# AGENTS.md — Vận hành HAgent (DeerFlow-AutoML)

## Mục tiêu

HAgent là trợ lý chat HAutoML. Runtime mặc định: **LangGraph multi-agent**
(hierarchy, campaign, plan executor, world model). OpenClaw chỉ legacy.

## Đọc theo thứ tự

1. `SOUL.md` — identity + quy tắc
2. `README_HAGENT.md` — kiến trúc Docker/runtime
3. `skills/hautoml/SKILL.md` — tools
4. `hagent.yaml` — LLM, planning, campaign, hierarchy, world_model

## Runtime Docker

```text
Frontend → HAgent Bridge (:9900)
              │ HAGENT_RUNTIME_MODE=deerflow (default)
              ▼
         toolkit (:8585)  /api/v1/chat/agent-run
              │
              ▼
         LangGraph (hagent.agent.graph)
              │ tools
              ▼
         HAutoML APIs + workers
```

Legacy:

```bash
HAGENT_RUNTIME_MODE=openclaw docker compose --profile openclaw up
```

## Phạm vi

- Chỉ HAutoML (dataset / train / job / predict / world state)
- Không tự viết code train
- Không hỏi JWT token

## File quan trọng

| Path | Vai trò |
|---|---|
| `chat_router.py` | FastAPI chat + `/agent-run` (DeerFlow) |
| `bridge/app.py` | JWT, conversation, forward deerflow/openclaw |
| `agent/graph.py` | Multi-agent orchestration |
| `agent/tools/automl_tools.py` | LangChain tools (source of truth) |
| `skills/hautoml/*` | OpenClaw CLI skill + docs |
| `world/*` | World model store/query |

## Khi cập nhật

- Thêm tool → `automl_tools.py` + `SKILL.md` + `tools.yaml`
- Đổi runtime → `docker-compose.yaml` + `README_HAGENT.md`
