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

## 2026-07-26 — TESTFIX-001

### Phạm vi

- Vá fixture `mock_llm_server` trong `src/backend/tests/conftest.py`.

### Quyết định

- Thay `except (httpx.ConnectError, httpx.ReadTimeout)` bằng
  `except httpx.TransportError` — trong httpx, `TimeoutException`
  (gồm `ConnectTimeout`, `ReadTimeout`) là subclass của `TransportError`,
  nên một except bao trùm mọi lỗi kết nối/timeout khi poll health.
- Bọc vòng poll bằng `try/finally` với cờ `ready`: fixture fail ở bất kỳ
  đường nào cũng `proc.kill()`, hết leak process chiếm port 11435.

### File thay đổi

- `src/backend/tests/conftest.py`, `feature_list.json`,
  `claude-progress.md`.

### Verification

- `pytest tests/test_deerflow_automl.py -m "not ollama"` — PASS (43 passed)
- `pytest tests -m "not ollama"` — PASS (207 passed, 0 failed)
- Port 11435 không còn listener sau suite — không leak.

### Rủi ro còn lại

- Không. Lỗi gốc chỉ tái hiện khi server khởi động chậm/timeout ở lần
  poll đầu (đã quan sát trên Windows lần chạy đầu tiên sau cài venv).

### Handoff

- HARNESS-001 vẫn `in_progress` (WIP 1/1). Tiếp theo theo kế hoạch:
  Giai đoạn 3 — outcome head, ensemble + calibration, benchmark layer.

## 2026-07-26 — WM-OUTCOME-001

### Phạm vi

- HARNESS-001 đóng hợp lệ (3 test_commands pass) trước khi mở task mới.
- Thêm `hagent/world/predictor/outcome_head_v1.py`: OutcomeHeadV1 (MLP 1 lớp
  ẩn, output [μ, s], σ = softplus(s)+1e-3), train Gaussian NLL grad-clip,
  `outcome_features` (one-hot algo/problem/metric + time_limit/models/dataset
  log-scaled + latent z tùy chọn), `extract_outcome_samples` từ trajectory
  docs (dedup theo job id), `rank_variants_by_outcome`.
- Factory `create_outcome_head` + export `__init__`; config
  `world_model.outcome_head` trong `hagent.yaml`.

### Verification

- `pytest tests/test_outcome_head.py` — PASS (24 passed)
- `pytest tests -m "not ollama"` — PASS (231 passed, 0 failed)

### Rủi ro còn lại

- Head chưa được nối vào campaign builder/compare (thuộc WM-CEM-001).
- σ aleatoric đơn lẻ chưa đủ cho calibration claim — cần ensemble
  (WM-ENSEMBLE-001 kế tiếp).

### Handoff

- Kế tiếp: WM-ENSEMBLE-001 (deep ensemble K seeds + world/calibration.py).

## 2026-07-26 — WM-ENSEMBLE-001

### Phạm vi

- `world/predictor/ensemble.py`: OutcomeEnsemble (K OutcomeHeadV1, seed lệch),
  mixture moments μ*, σ*² (aleatoric + epistemic), save/load thư mục
  member_{i}.npz, `train_outcome_ensemble`.
- `world/calibration.py`: interval_coverage, ECE (PIT), reliability_table,
  sharpness — stdlib NormalDist, không thêm dependency.
- Factory `create_outcome_ensemble`, config `world_model.outcome_ensemble`.

### Verification

- `pytest tests/test_world_model_calibration.py` — PASS (20 passed)
- `pytest tests -m "not ollama"` — PASS (251 passed, 0 failed)

### Handoff

- Kế tiếp: WM-SURPRISE-001 (outcome-space surprise + nối wm_hooks).

## 2026-07-26 — WM-SURPRISE-001

### Phạm vi

- `world/surprise.py`: compute_outcome_surprise (z-score |y−μ|/σ, ngưỡng
  outcome_thresholds riêng, σ floor 1e-6).
- `wm_hooks.py`: campaign_outcome_surprise (+_default_outcome_model: ensemble
  → head fallback từ config).
- `runner.py`: phát event `campaign_outcome_surprise` đúng lúc variant chuyển
  completed, gate `world_model.surprise.outcome_enabled`.

### Verification

- `pytest tests/test_outcome_surprise.py` — PASS (13 passed)
- `pytest tests -m "not ollama"` — PASS (264 passed, 0 failed)

### Handoff

- Kế tiếp: WM-CEM-001 (CEM thật trên config space + nối outcome ranking vào
  builder).

## 2026-07-26 — WM-CEM-001

### Phạm vi

- `world/planner/cem_config_v1.py`: CEM đúng nghĩa trên
  {search_algorithm × time_limit} — categorical sampling, elite refit với
  Laplace smoothing, score μ+βσ, cache, deterministic theo seed; fallback
  round-robin khi model chưa sẵn sàng.
- `builder.py`: slot trống được đề xuất bởi planner (source `wm_planner`,
  gate `campaign.wm_variant_proposal`), toàn bộ variant xếp thứ tự submit
  theo mean dự đoán (gate `campaign.wm_rank_variants`); hành vi cũ giữ
  nguyên khi không có checkpoint model.
- Factory `create_campaign_planner`; config `world_model.campaign_planner`.

### Verification

- `pytest tests/test_cem_config_planner.py` — PASS (14 passed)
- `pytest tests -m "not ollama"` — PASS (278 passed, 0 failed)

### Rủi ro còn lại

- Lưu ý test: `data/memory/` sót fact warm-start từ lần chạy harness smoke
  làm builder sinh variant warm_start cho user u1 — hành vi cũ hợp lệ nhưng
  test mới không nên assert cứng danh sách source.

### Handoff

- Kế tiếp: BENCH-001 (tầng benchmark score-vs-jobs/regret + script + CI).

## 2026-07-26 — BENCH-001

### Phạm vi

- `agent/eval/benchmark.py`: DatasetProfile (response surface biết trước →
  đo regret so optimum), 3 profiles synth_strong/noisy/flat;
  SimulatedAutoMLEnv cắm qua set_tool_invoker; run_condition với 4 loại
  điều kiện (wm train online giữa campaigns / no_wm / random / fixed_<algo>);
  run_benchmark_matrix.
- `agent/eval/metrics.py`: best_so_far_curve, jobs_to_threshold,
  normalized_regret, aggregate_curves.
- `scripts/run_benchmark.py`: CLI ma trận + aggregate + JSON + bảng tóm tắt.

### Verification

- `pytest tests/test_benchmark.py` — PASS (17 passed)
- `pytest tests -m "not ollama"` — PASS (295 passed, 0 failed)
- CLI smoke 4 điều kiện × 3 seeds × budget 12: wm 0.8896 > random 0.8790 >
  fixed_grid 0.6934 (synth_strong).

### Rủi ro còn lại

- Không gian config hiện nhỏ (3 algo × 3 time) nên wm vs no_wm sát nhau ở
  smoke — thí nghiệm thật cần mở rộng không gian (models subset, budget lớn)
  để tách biệt rõ.
- CI benchmark.yml chưa tạo: `.github/**` thuộc protected_paths, cần user
  phê duyệt trước khi thêm workflow.

### Handoff

- Kế tiếp: USAGE-001 (token/cost tracker) rồi DATA-OPENML-001 (fetch script).

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
