# Giải Thích HAgent Trong HAutoML

Tài liệu này giải thích HAgent theo đúng code hiện có trong repo: HAgent là
gì, nằm ở đâu trong hệ thống, và một tin nhắn chat đi qua những lớp nào
trước khi thành kết quả trên giao diện.

## 1. HAgent là gì?

HAgent là trợ lý AI nhúng trong HAutoML, chạy bằng **LangGraph multi-agent**
tích hợp thẳng vào backend (không có process agent riêng bên ngoài). Nó
không tự huấn luyện model hay xử lý dữ liệu bằng code riêng. Vai trò:

- Nhận yêu cầu người dùng qua chat widget.
- Hiểu ý định: liệt kê dataset, xem feature, train model, xem job…
- Điều phối multi-agent: phân rã mục tiêu (hierarchy), chạy chiến dịch
  nhiều ứng viên (campaign), gọi tool HAutoML.
- Trả kết quả rõ ràng bằng tiếng Việt.

```text
Người dùng hỏi bằng ngôn ngữ tự nhiên
        |
        v
LangGraph graph (coordinator → hierarchy/campaign → synthesize)
        |
        v
LangChain tools gọi REST API của HAutoML
        |
        v
HAgent tóm tắt kết quả cho người dùng
```

## 2. Thành phần

| Thành phần | File / thư mục | Vai trò |
|---|---|---|
| Chat UI | `frontend/src/components/chatWidget/ChatWidget.tsx` | Giao diện chat |
| Chat client | `frontend/src/api/chatClient.ts` | Gọi API chat, upload, health, conversation |
| Next proxy | `frontend/src/app/api/hagent/[...path]/route.ts` | Forward frontend → HAgent Bridge |
| HAgent Bridge | `backend/hagent/bridge/app.py` | Auth JWT, lưu hội thoại, forward tới agent runtime |
| Conversation store | `backend/hagent/bridge/conversation.py` | Lịch sử chat trong MongoDB |
| Chat router | `backend/hagent/chat/router.py` | Endpoint `/api/v1/chat/*` + `/agent-run` trên toolkit |
| Agent graph | `backend/hagent/agent/graph.py` | LangGraph StateGraph — điều phối multi-agent |
| Tools | `backend/hagent/agent/tools/automl_tools.py` | HAutoML API wrappers (source of truth) |
| LLM config | `backend/hagent/agent/llm_config.py` | Đa provider, resolve strict, per-request model |
| World model | `backend/hagent/world/` | Trạng thái + dự đoán outcome + surprise |
| Campaign | `backend/hagent/agent/campaign/` | Chiến dịch train nhiều ứng viên, WM ranking |

## 3. Đường đi của một tin nhắn

1. Người dùng gõ tin nhắn trong ChatWidget; frontend gửi kèm JWT qua Next
   proxy tới **Bridge** (`:9900`).
2. Bridge xác thực JWT, lưu tin nhắn vào MongoDB, rồi POST tới toolkit
   `POST /api/v1/chat/agent-run` (URL override bằng `HAGENT_AGENT_RUN_URL`).
3. `hagent.chat.router` trên toolkit kiểm tra model theo request (`{"model": ...}`
   — tên sai trả HTTP 400) rồi chạy **LangGraph graph**: coordinator phân
   loại ý định → hierarchy phân rã mục tiêu → campaign/plan executor chạy
   tool → synthesizer viết câu trả lời.
4. Tool gọi REST API HAutoML (dataset/train/job); world model cập nhật
   trạng thái, chấm surprise, có thể mở rộng campaign khi kết quả lệch
   dự đoán (event `campaign_extended`).
5. Kết quả (message + `cost_metrics` tokens/USD + campaign/hierarchy
   status) trả ngược Bridge → frontend; Bridge lưu câu trả lời.

## 4. Tool ↔ API HAutoML

`automl_tools.py` là nơi agent thật sự chạm vào HAutoML API:

| Tool | API HAutoML |
|---|---|
| `health` | `GET /home` |
| `list_datasets` | `POST /get-list-data-by-userid` |
| `get_dataset_info` | `GET /get-data-info` |
| `get_features` | `GET /v2/auto/features` |
| `preview_data` | `GET /v2/auto/data` |
| `get_available_models` | `GET /api/v1/available-models/{problem_type}` |
| `get_metrics` | `GET /v2/auto/metrics` |
| `start_training` | `POST /v2/auto/jobs/training` |
| `list_jobs` | `POST /get-list-job-by-userId` |
| `get_job_info` | `POST /get-job-info` |
| `batch_predict` | `POST /v2/auto/{job_id}/predictions` |
| `delete_dataset` | `DELETE /delete-dataset/{dataset_id}` |

## 5. Cấu hình

Tất cả từ `backend/hagent/config/hagent.yaml` (env `HAGENT_CONFIG` trỏ file khác
khi cần). Không hard-code trong Python.

| Env | Mặc định | Ý nghĩa |
|---|---|---|
| `HAGENT_CONFIG` | `hagent/config/hagent.yaml` | File cấu hình trung tâm |
| `HAGENT_AGENT_RUN_URL` | `{HAUTOML}/api/v1/chat/agent-run` | Bridge → toolkit |
| `LLM_DEFAULT_MODEL` | (yaml) | Model mặc định — resolve strict, sai tên là lỗi ngay |
| `OLLAMA_BASE_URL` | host gateway | LLM local |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `META_AI_API_KEY` | — | Key provider tương ứng |

LLM đa provider khai báo trong `hagent.yaml llm.models` (openai, anthropic,
ollama, openai_compatible). Client chọn model theo từng request bằng
`{"model": "<tên>"}`; usage (tokens + USD) trả về trong `cost_metrics`.

## 6. Các Endpoint Chat Chính

| Method | Endpoint | Vai trò |
|---|---|---|
| `POST` | `/api/v1/chat/` | Gửi tin nhắn chat |
| `POST` | `/api/v1/chat/upload` | Gửi tin nhắn kèm file |
| `GET` | `/api/v1/chat/health` | Kiểm tra Bridge, agent runtime và HAutoML backend |
| `GET` | `/api/v1/chat/suggestions` | Lấy gợi ý ban đầu |
| `GET` | `/api/v1/chat/conversations` | Danh sách hội thoại gần đây |
| `GET` | `/api/v1/chat/conversation/{conversation_id}` | Toàn bộ message của một hội thoại |
| `DELETE` | `/api/v1/chat/conversation/{conversation_id}` | Xóa hội thoại |
| `GET` | `/api/v1/chat/providers` | Liệt kê provider khả dụng |

## 7. Chạy bằng Docker

```bash
cd src/backend
docker compose up --build -d          # toolkit + bridge + mongo + kafka + minio
docker compose --profile worker up -d # thêm worker
```

- MỘT image backend (`hautoml.toolkit.dockerfile`) dùng chung
  toolkit/worker/nano; bridge có image mỏng riêng (`hagent/bridge/Dockerfile`).
- Health: bridge `GET :5360/api/v1/chat/health`, toolkit `GET :5370/home`.
- Chi tiết deploy đầy đủ: `docs/deploy.md` (repo root).

## 8. Tóm Tắt Dễ Nhớ

```text
HAgent = Chat UI + Bridge + LangGraph multi-agent + LangChain tools + World model
```

- HAgent không thay thế HAutoML backend — chỉ là lớp trợ lý điều phối.
- Bridge là lớp API chat quan trọng nhất: auth, history, upload, health,
  polling training result.
- `automl_tools.py` là nơi agent thật sự chạm vào HAutoML API.
- MongoDB lưu lịch sử hội thoại và kết quả assistant.

## 9. Ghi chú công nghệ

Kiến trúc multi-agent của HAgent xây trên **LangGraph** (StateGraph, tool
calling, SSE streaming), tham khảo các mẫu điều phối lead-agent/sub-agent
hiện đại; toàn bộ mã tích hợp nằm trong `backend/hagent/` của dự án.
