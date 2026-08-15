# AGENTS.md — Vận hành HAgent

## Mục tiêu

HAgent là trợ lý chat HAutoML. Runtime: **LangGraph multi-agent**
(hierarchy, campaign, plan executor, world model) — xây dựng trên công nghệ
multi-agent của LangGraph, tích hợp thẳng vào backend HAutoML.

## Đọc theo thứ tự

1. `README_HAGENT.md` — kiến trúc Docker/runtime
2. `config/hagent.yaml` — LLM, planning, campaign, hierarchy, world_model
3. `agent/tools/automl_tools.py` — danh sách tool (source of truth)

## Runtime Docker

```text
Frontend → HAgent Bridge (:9900)
              │
              ▼
         toolkit (:8585)  /api/v1/chat/agent-run
              │
              ▼
         LangGraph (hagent.agent.graph)
              │ tools
              ▼
         HAutoML APIs + workers
```

## Phạm vi

- Chỉ HAutoML (dataset / train / job / predict / world state)
- Không tự viết code train
- Không hỏi JWT token

## File quan trọng

| Path | Vai trò |
|---|---|
| `chat/router.py` | FastAPI chat + `/agent-run` (LangGraph) |
| `bridge/app.py` | JWT, conversation, forward tới agent runtime |
| `agent/graph.py` | Multi-agent orchestration |
| `agent/tools/automl_tools.py` | LangChain tools (source of truth) |
| `world/*` | World model store/query |

## Khi cập nhật

- Thêm tool → `automl_tools.py`
- Đổi runtime → `docker-compose.yaml` + `README_HAGENT.md`
