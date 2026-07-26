# Agent Progress Log

## Trạng thái hiện tại

- Task: `HARNESS-001` — Thiết lập Minimal Agent Harness
- Trạng thái: `in_progress`
- WIP: `1/1`
- Bước tiếp theo: chạy đủ `test_commands`, review phạm vi rồi mới đánh dấu
  `done`.

## 2026-07-24 — HARNESS-001

### Phạm vi

- Tạo `AGENTS.md`, `init.sh`, `feature_list.json`, `claude-progress.md`.
- Không sửa module nghiệp vụ trong `src/`.

### Quyết định

- Dùng `feature_list.json` làm nguồn sự thật cho WIP, whitelist và bằng chứng
  kiểm thử.
- Khóa file nhạy cảm, auth, dependency, migration và lõi Agent theo mặc định.
- `init.sh` chỉ kiểm tra sức khỏe và trạng thái; không tự cài dependency, xóa
  file hoặc thay đổi source.

### Verification

- Trạng thái: pending.
- Chưa được kết luận hoàn thành cho đến khi mọi lệnh test trả về mã thoát `0`.

### Handoff

- Chưa bàn giao; task đang chờ kiểm thử.

## 2026-07-26 — GIT-SETUP-001

### Phạm vi

- Phát hiện cây `src/` bị Google Drive sync làm rỗng (~1273/1435 file 0 byte);
  không còn nội dung local để merge.
- Lưu `recovery-manifest.txt` (tên + size + mtime của toàn bộ cây rỗng) phục vụ
  khôi phục Google Drive version-history (cửa sổ ~30 ngày từ 24/07).
- Backup 4 file harness + manifest ra `D:/Homeworks/nckh-hollow/` và
  `C:/Users/Admin/Desktop/nckh-harness-backup-20260726/`.
- Cách ly `src/backend`, `src/frontend`, `src/tools` sang
  `D:/Homeworks/nckh-hollow/src/` (rename, không xóa; giữ ≥30 ngày).
- `git init` tại gốc; `core.autocrlf=input`, `core.longpaths=true`;
  remote `origin` = VanAnh-13/test, `upstream` = optivisionlab/AutoML
  (push URL DISABLED); fetch; tạo nhánh `hagent` từ
  `origin/features/deerflow-automl` @ `7a397ad`.

### Quyết định

- Checkout code từ remote không tính là "edit" theo whitelist — task này chỉ
  sửa `feature_list.json`, `claude-progress.md` (control files) và tạo
  `recovery-manifest.txt` (giữ local qua `.git/info/exclude` vì repo public).
- Nested `.git` rỗng (0 commit, không remote) trong `src/backend` cũ đi theo
  cây quarantine, không đụng vào `.git/**` của repo mới.
- Không đụng clone không liên quan tại `D:/Homeworks/python/nckh`.

### File thay đổi

- `feature_list.json` (thêm task GIT-SETUP-001, done), `claude-progress.md`
  (mục này), `recovery-manifest.txt` (mới, local-only).

### Verification

- `python -m py_compile src/backend/app.py` — PASS (0)
- `bash init.sh` — PASS (0) — chạy sau khi cập nhật bookkeeping
- `git status --porcelain` — sạch sau commit bookkeeping

### Rủi ro còn lại

- Google Drive đang TẮT; sau khi bật lại phải chờ sync ổn định rồi chạy
  `git status --porcelain` — file bị Drive làm rỗng sẽ hiện modified; sửa bằng
  `git restore .` (nguồn khôi phục: GitHub `origin/hagent` sau khi push).
- Thế hệ code mới hơn (world_model deep ensemble, CEM-MPC, shadow monitor,
  MCP) mất — user chọn rebuild phần cần cho bài báo ACML 2026 workshop.

### Handoff

- Tiếp theo: push `hagent` lên origin; dựng môi trường (deps + docker +
  Ollama); phát triển theo kế hoạch bài báo (outcome head → ensemble →
  benchmark). HARNESS-001 vẫn `in_progress` (WIP 1/1).

## 2026-07-26 — Baseline môi trường (Giai đoạn 2, không sửa source)

### Phạm vi

- Venv `src/backend/.venv` (Python 3.12.13, uv 0.11.30) + requirements.txt
  + dev deps (pytest 9.1.1, pytest-asyncio, pytest-timeout).
- Không sửa file source nào; chỉ cài đặt môi trường (gitignored).

### Verification

- `pytest tests -m "not ollama" --timeout=120` — PASS: **207 passed,
  7 deselected (ollama), 0 failed** (6.42s).
- `python scripts/run_agent_harness.py --layer offline,graph
  --modes single_shot,plan_executor,campaign,hierarchical --tags smoke`
  — PASS: 16/16 OK (gồm wm_human_train_glass world-model scenarios).
- Docker 29.6.2 sẵn sàng; Ollama CHƯA cài (cần trước khi chạy thí nghiệm LLM
  local qwen2.5:14b).

### Rủi ro còn lại

- Flake môi trường: fixture `mock_llm_server` trong `tests/conftest.py` không
  bắt `httpx.ConnectTimeout` khi poll `/health` → lần chạy đầu lỗi 5 test và
  leak process chiếm port 11435 (đã kill PID sót). Chạy lại sạch. Cần task
  riêng để vá except-tuple này.

## Mẫu ghi cho phiên tiếp theo

```text
## YYYY-MM-DD — TASK-ID

### Phạm vi
- ...

### Quyết định
- ...

### File thay đổi
- ...

### Verification
- `command` — PASS/FAIL (mã thoát)

### Rủi ro còn lại
- ...

### Handoff
- ...
```
