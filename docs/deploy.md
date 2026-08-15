# Deploy HAgent (LangGraph multi-agent + World Model)

Quy trình dựng full stack bằng Docker Compose, bao gồm cả bước bật world
model (checkpoint) và cơ chế surprise-driven replanning.

## 0. Yêu cầu

- Docker + Docker Compose v2.
- LLM: **một trong** (a) Ollama trên host (`ollama pull qwen2.5:14b`,
  khuyến nghị `OLLAMA_KEEP_ALIVE=-1`), (b) API key OpenAI/Anthropic,
  (c) server OpenAI-compatible tự host.
- RAM ≥ 16GB nếu chạy qwen2.5:14b bằng CPU.

## 1. Cấu hình môi trường

```bash
cd src/backend
cp deploy.llm.env.example .env    # rồi điền SECRET_KEY, MINIO_*, API keys
```

Lưu ý **resolve model strict**: `LLM_DEFAULT_MODEL` sai tên sẽ làm request
LLM đầu tiên lỗi ngay kèm danh sách tên hợp lệ — kiểm tra trước bằng:

```bash
python -c "from hagent.agent.llm_config import get_default_model_config as g; print(g())"
```

## 2. (Tùy chọn nhưng khuyến nghị) Bật world model

Đường "auto" của campaign builder/runner tự nạp checkpoint từ
`data/world_model/outcome_head_v2.npz` + `outcome_ensemble_v2/`. Chưa có
checkpoint thì agent vẫn chạy bình thường, chỉ là các gate WM bất hoạt.

```bash
# Từ Mongo production (world_trajectories):
python scripts/train_outcome_model.py --source mongo

# Hoặc từ file JSONL (artifact tái lập được):
python scripts/train_outcome_model.py --source jsonl --jsonl data/traj.jsonl
```

Ghi lại `head_sha256` mà script in ra — mọi kết quả thí nghiệm nên kèm SHA
này. **Sau khi train, KHÔNG đổi vocab trong hagent.yaml** (`outcome_head` /
`outcome_ensemble.search_algorithms`, `model_vocab`) — đổi vocab là đổi
chiều feature, checkpoint cũ vô hiệu (path đã version v2 để chặn nạp nhầm).

Bật cơ chế replanning (điều kiện thí nghiệm C) trong `hagent/config/hagent.yaml`:

```yaml
agent:
  campaign:
    surprise_extension:
      enabled: true
```

## 3. Khởi động stack

**Kiến trúc image (sau lần viết lại 27/7/2026):** MỘT image backend
`hautoml-toolkit` (build từ `hautoml.toolkit.dockerfile`, python:3.12-slim)
dùng chung cho toolkit / worker / nano — chỉ khác `command`; bridge có image
mỏng riêng (`hagent/bridge/Dockerfile`). `worker.dockerfile` và
`hautoml.nano.dockerfile` đã bỏ — không cần build tay `workers:latest` nữa:

```bash
docker compose --profile worker up --build -d
```

Service nào cũng có healthcheck; chờ tất cả `healthy`:

```bash
docker compose ps
```

| Service | Cổng host | Health |
|---|---|---|
| toolkit (LangGraph agent + API) | 5370 | `GET /home` |
| hagent_bridge | 5360 | `GET /api/v1/chat/health` |
| mongo | 5380 | `mongosh ping` |
| minio | 5381 | `mc ready local` |
| kafka | 5383 | (sẵn có) |

Bridge `depends_on` toolkit ở mức `service_healthy` — bridge chỉ lên khi
agent thật sự sẵn sàng.

## 4. Smoke test sau deploy

```bash
# 1. Health + danh sách model
curl -s http://localhost:5370/api/v1/chat/models

# 2. E2E có auth (signup → login → upload → chat → poll job)
python scripts/e2e_docker_test.py

# 3. Harness API layer (BẮT BUỘC --require-live: không có nó, stack chết
#    vẫn báo xanh)
python scripts/run_agent_harness.py --layer api \
  --base-url http://localhost:5360 --token "$JWT" --require-live --json api.json
```

## 5. Chọn model theo request

```bash
curl -s -X POST http://localhost:5370/api/v1/chat/agent-run \
  -H "Authorization: Bearer $JWT" -H "Content-Type: application/json" \
  -d '{"message": "train my dataset ds1, target y", "model": "openai-gpt4o-mini"}'
```

Tên model sai → HTTP 400 kèm danh sách hợp lệ. `cost_metrics` trong response
chứa tokens + USD (bảng giá: `hagent.yaml llm.usage_tracking.pricing`).

## 6. Vận hành

- **Retrain checkpoint**: chạy lại script ở bước 2 — `_default_outcome_model`
  memoize theo mtime nên container tự nhặt checkpoint mới, không cần restart.
- **Log**: mọi service giới hạn json-file 10MB×3.
- **Kafka worker**: profile `worker`; thêm worker bằng cách nhân bản service
  với `WORKER_INDEX` khác.
- **Secret không vào image**: `.dockerignore` chặn `.env*` — key chỉ đi
  vào container qua `environment:` của compose (đọc từ `.env` lúc `up`).
  Model `meta-ai` cần `META_AI_API_KEY` trong `.env` (đã passthrough sẵn
  trong compose).
