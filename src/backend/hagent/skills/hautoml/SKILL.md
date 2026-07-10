---
name: hautoml
description: >-
  HAutoML platform tools for DeerFlow-AutoML / HAgent: datasets, features,
  multi-candidate training, jobs, predictions, world state. Prefer structured
  LangChain tools in DeerFlow runtime; CLI exec remains for OpenClaw legacy.
metadata:
  {
    "openclaw": { "emoji": "📦", "requires": { "bins": ["python3"] } },
    "deerflow": {
      "runtime": "langgraph",
      "tools_module": "hagent.agent.tools.automl_tools"
    }
  }
---

# Skill HAutoML (DeerFlow-AutoML)

Điều khiển nền tảng **HAutoML** cho user đã đăng nhập.

## Runtime

| Mode | Cách gọi tool |
|---|---|
| **DeerFlow** (mặc định Docker) | LangGraph tools trong `automl_tools.py` — **không** cần `exec` |
| **OpenClaw** (legacy profile) | `exec` + CLI `hautoml_tools.py` bên dưới |

**Không viết code train.** Chỉ gọi tool/API HAutoML.

## Xác thực

Bridge/toolkit bơm `USER_TOKEN` + `USER_ID`. **Không hỏi token.**

## World Model

Ưu tiên snapshot world state (datasets/jobs/plans) trước khi gọi tool lặp.
Sau train: gợi ý campaign multi-job / hierarchy analyze→train→evaluate khi phù hợp.

## DeerFlow LangChain tools (primary)

| Tool | Khi nào |
|---|---|
| `list_datasets` | Liệt kê dataset |
| `get_dataset_info` | Chi tiết 1 dataset |
| `get_features` | Cột / features |
| `preview_data` | Xem trước rows |
| `get_available_models` | Thuật toán theo problem_type |
| `get_metrics` | Metric đánh giá |
| `start_training` | Submit job (metric, search_algorithm, time_limit) |
| `list_jobs` / `get_job_info` | Theo dõi job |
| `cancel_job` | Hủy job |
| `predict_batch` | Batch predict (job đã train) |
| `get_world_state` | Snapshot WM |
| `check_system_health` | Health HAutoML |

Agent graph (không phải skill CLI): **hierarchy**, **campaign** (N configs), **plan_executor**, **reviser**.

## OpenClaw CLI (legacy)

Đường dẫn:

```text
/app/hagent/skills/hautoml/scripts/hautoml_tools.py
```

hoặc trong container OpenClaw:

```text
/home/node/.openclaw/skills/hautoml/scripts/hautoml_tools.py
```

Ví dụ:

```bash
python3 /app/hagent/skills/hautoml/scripts/hautoml_tools.py list_datasets \
  --user-id "$USER_ID" --token "$USER_TOKEN"
```

```bash
python3 /app/hagent/skills/hautoml/scripts/hautoml_tools.py start_training \
  --dataset-id "<DATASET_ID>" \
  --problem-type "classification" \
  --target-column "<TARGET>" \
  --features "f1,f2,f3" \
  --metric-sort "f1" \
  --search-algorithm "grid_search" \
  --max-time 300 \
  --user-id "$USER_ID" --token "$USER_TOKEN"
```

```bash
python3 /app/hagent/skills/hautoml/scripts/hautoml_tools.py batch_predict \
  --job-id "<JOB_ID>" --file-path "/path/data.csv" --token "$USER_TOKEN"
```

```bash
python3 /app/hagent/skills/hautoml/scripts/hautoml_tools.py cancel_job \
  --job-id "<JOB_ID>" --token "$USER_TOKEN"
```

```bash
python3 /app/hagent/skills/hautoml/scripts/hautoml_tools.py get_world_state \
  --user-id "$USER_ID" --token "$USER_TOKEN"
```

## Nguyên tắc trả lời

1. Tiếng Việt (trừ khi user dùng ngôn ngữ khác).
2. Bảng Markdown cho list dataset/job; kèm ID.
3. Gợi ý bước tiếp theo (analyze → train → evaluate → predict).
4. Lỗi `401` → đăng nhập lại; `404` → liệt kê lại; timeout → thử lại.

## Lỗi thường gặp

| Hiện tượng | Gợi ý |
|---|---|
| 401 | Phiên hết hạn |
| Missing target/features | Gọi `get_features` / xác nhận cột target |
| Job pending lâu | `get_job_info` / worker Kafka |
| OpenClaw path not found | Đang ở DeerFlow — dùng LangChain tools, không `exec` |
