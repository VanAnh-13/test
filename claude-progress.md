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

## 2026-07-26 — USAGE-001 + DATA-OPENML-001 (hoàn tất Giai đoạn 3)

### Phạm vi

- USAGE-001: `agent/middlewares/usage_tracker.py` — UsageTracker (thread-safe,
  cost per-1M-token từ `llm.usage_tracking.pricing`), UsageTrackingCallback
  (parse token_usage lẫn usage_metadata), factory.
- DATA-OPENML-001: `scripts/fetch_openml_datasets.py` — registry 8 datasets
  theo data_id cố định, normalize_frame, CLI --list/--datasets/--out, ghi
  CSV + manifest.json vào assets/openml/. Chưa tải dữ liệu thật (cần mạng).

### Verification

- `pytest tests/test_usage_tracker.py` — PASS (12), `tests/test_fetch_openml.py`
  — PASS (5), CLI --list — PASS.
- Full suite cuối: **312 passed, 0 failed** (từ 207 baseline → +105 test mới).

### Trạng thái Giai đoạn 3 (kế hoạch ACML): 7/7 hạng mục DONE

1. outcome_head_v1 ✓  2. ensemble + calibration ✓  3. outcome surprise ✓
4. CEM config planner + builder ✓  5. benchmark layer + CLI ✓
6. usage tracker ✓  7. OpenML fetcher ✓

### Còn chờ bên ngoài

- Tải OpenML data + cài Ollama trước khi chạy thí nghiệm thật (Giai đoạn 4).
- CI benchmark.yml cần user phê duyệt (.github/** protected).
- Danh sách workshop ACML 2026 (~07/08) → chốt venue + page limit.

## 2026-07-26 — REVIEW-FIX-001 (sau adversarial review 19 agents)

### Phạm vi — 6 lỗi xác nhận, đều đã sửa

- A (critical, train/serve skew): builder truyền `dataset_meta` từ world_model
  snapshot vào CEM planner + ranking; `use_latent: false` mặc định cho
  outcome_head/ensemble (z không có tại thời điểm build campaign).
- B (critical, metric): benchmark tính curve/regret/jobs_to_95 trên
  EXPECTED score của config đã submit (điểm nhiễu báo cáo riêng
  `observed_*`) — hết bias max-order-statistic trên profile nhiễu.
- C (critical, contamination): `collect_warm_start_configs(top_k<=0)` → []
  (tắt cả nguồn memory); benchmark dùng user_id riêng theo condition.
- D (major): `build_campaign`/`campaign_step` nhận `outcome_model` sentinel
  "auto"/None/model — None TẮT hẳn fallback checkpoint đĩa; benchmark truyền
  model online vào runner nên `n_outcome_surprise_events` đo đúng model.
- E (major): CEM acquisition sửa thành `sign*μ + β·σ` (optimism đúng cả khi
  minimize).
- F (major): validate conditions/profiles TRƯỚC khi chạy ma trận (CLI +
  run_benchmark_matrix) — typo không đốt kết quả đã chạy.

### Verification

- 3 file test liên quan: 55 passed; full suite: **323 passed, 0 failed**.
- CLI synth_noisy: fixed_grid regret 0.6667 (đúng bản chất), wm regret 0.0.

### Handoff

- Giai đoạn 3 + review hardening HOÀN TẤT. Sẵn sàng Giai đoạn 4 (thí nghiệm
  thật): tải OpenML (`scripts/fetch_openml_datasets.py`), cài Ollama, mở rộng
  config space, chạy ma trận chính thức.

## 2026-07-26 — WM-SPACE-001 (hướng A mở rộng world model)

### Phạm vi

- CEM planner: chiều model-subset (Bernoulli per-model, elite refit smoothing,
  clip [0.02,0.98], min_models) + categorical_dims tùy config (vd. cv_folds);
  model_options rỗng → hành vi cũ.
- outcome_features: multi-hot model_vocab (lưu/khôi phục checkpoint; vocab
  rỗng giữ nguyên chiều cũ).
- builder: campaign config override search space của planner (kể cả []).
- Benchmark: DatasetProfile.model_effects + dilution; profile synth_models
  (3 algo × 3 time × 2^4 subset); random condition sample subset; hagent.yaml
  khai 6 model catalog thật (assets/system_models/classification.yml).

### Verification

- 74 passed (3 file test), full suite **333 passed**.
- CLI synth_models (budget 18 × 3 seeds): wm 0.7686 > no_wm 0.7257 —
  không gian lớn tách rõ steering; random 0.7727 nhỉnh hơn wm ở budget nhỏ
  → explore/exploit scheduling là việc của WM-MPC-001 (hướng C).

### Handoff

- Kế tiếp: WM-META-001 (hướng B — meta-features giàu + transfer LOO).

## 2026-07-26 — WM-META-001 (hướng B mở rộng world model)

### Phạm vi

- `world/meta_features.py`: META_KEYS_V2 + meta_features_from_frame (tính từ
  DataFrame thật — dùng được cho OpenML CSV).
- `outcome_features` meta_profile v2: 7 meta keys + **giao chéo tường minh
  algo_onehot × [log_rows, frac_categorical]** (SGD thuần không tự học được
  interaction — đã chứng minh bằng thí nghiệm); checkpoint cũ fallback v1.
- `train_outcome_head`: **warmup MSE** (epochs//3) trước Gaussian NLL — NLL
  từ đầu làm σ phình sớm, μ sập về mean.
- Benchmark: make_transfer_profiles (best algo phụ thuộc meta theo luật cố
  định), generate_offline_samples, run_transfer_loo (pretrained vs scratch),
  run_condition nhận initial_samples (pretrain → có model từ job 0).

### Verification

- `pytest tests/test_meta_features.py` — PASS (11); full suite **344 passed**.
- Transfer thật: head học 5 profile → đoán đúng thuật toán tốt nhất trên
  profile CHƯA THẤY; LOO pretrained ≥ scratch cùng seed/budget.

### Handoff

- Kế tiếp: WM-MPC-001 (hướng C — budget-aware batch planning, receding
  horizon; giải quyết random > wm ở budget nhỏ quan sát tại WM-SPACE-001).

## 2026-07-26 — WM-MPC-001 (hướng C mở rộng world model)

### Phạm vi

- `world/planner/cem_mpc_v1.py`: CemMpcV1Planner — pool ứng viên từ CEM,
  chọn batch bằng Thompson sampling với σ_eff = σ·√(remaining_after/total);
  batch cuối exploit thuần; tương thích interface builder; factory backend
  `cem_mpc_v1`.
- Benchmark: condition `wm_mpc` (re-plan mỗi campaign với model mới nhất =
  receding horizon), train online như wm.

### Verification

- 34 passed (2 file), full suite **352 passed**.
- CLI synth_models (budget 18 × 3 seeds): **wm_mpc 0.7900, regret 0.0 —
  đạt optimum, jobs95=10.3** > random 0.7727 > wm 0.7552 > no_wm 0.7257.
  Budget-annealed exploration giải đúng vấn đề "random > wm ở budget nhỏ".

### Handoff

- Kế tiếp: WM-DYN-001 (hướng D — ensemble transition model + latent surprise
  chuẩn hóa theo σ).

## 2026-07-26 — WM-DYN-001 (hướng D — HOÀN TẤT lộ trình A+B+C+D)

### Phạm vi

- `world/predictor/dynamics_ensemble.py`: K NeuralJepaV1 seed lệch; predict
  trả mean latent + meta.std per-dim; save/load member_i.npz; factory backend
  `dynamics_ensemble` (tương thích WorldPredictor — cắm được vào service).
- `world/surprise.py`: compute_normalized_latent_surprise (RMS z-score
  per-dim, ngưỡng normalized_thresholds đơn vị z, sigma_floor);
  compute_surprise TỰ PHÁT HIỆN meta.std → mọi call site hưởng lợi không sửa.
- Untrained ensemble → identity, KHÔNG bịa std.

### Verification

- 12 passed (test mới); full suite **364 passed, 0 failed**.

### Tổng kết mở rộng world model (A+B+C+D, 26/7)

- A: không gian hành động 3×3 → 3×3×2^k (model subsets, Bernoulli-CEM).
- B: meta-features v2 + giao chéo algo×meta + warmup NLL → transfer LOO
  hoạt động (đoán đúng algo tốt nhất trên dataset chưa thấy).
- C: CEM-MPC budget-annealed Thompson → wm_mpc ĐẠT OPTIMUM synth_models
  (regret 0.0) vượt random/wm/no_wm.
- D: dynamics ensemble + surprise chuẩn hóa theo uncertainty.
- Suite: 207 (baseline) → **364 passed**. Điều kiện benchmark: wm, wm_mpc,
  no_wm, random, fixed_<algo>; + transfer LOO protocol.

## 2026-07-26 — HPO-IMPROVE-001 (cải thiện thuật toán + tốc độ HPO)

### Phạm vi

- **2 bug BO sửa**: (1) check hội tụ dừng ngay iteration đầu không cải thiện
  sau patience (abs(0)<threshold) → giờ yêu cầu patience bước liên tiếp
  (`_converged`, có unit test tái hiện bug cũ); (2) crash khi scoring=None.
- **BO dimension inference** (`infer_dimensions`, mặc định bật): list số
  nguyên → Integer(min,max); list thực → Real (log-uniform khi span ≥100×);
  mixed/bool/chuỗi → Categorical như cũ. GP giờ tối ưu trên không gian
  liên tục thật thay vì chọn giữa vài điểm rời rạc.
- **RandomSearchStrategy mới** (baseline chuẩn, dedup, enumerate-all khi
  grid nhỏ) + **SuccessiveHalvingStrategy mới** (multi-fidelity theo fraction
  dữ liệu, eta=3, stratified subsample, resource_frac trong cv_results).
- **Tốc độ (yêu cầu bổ sung)**: GA bỏ oversubscription lồng nhau (inner
  cross_validate n_jobs=-1 TRONG Parallel → n² tiến trình) + backend loky
  thay threading; Grid backend selector hết chọn threading (GIL-bound) cho
  workload trung bình; BO thêm batch ask/tell constant-liar
  (batch_size='auto' theo n_jobs) đánh giá lô điểm song song; Random/SH
  đánh giá ứng viên song song chunked (inner cv=1).
- Factory đăng ký random*/successive*/halving/sh + alias.

### Verification

- `pytest tests/test_search_strategies.py` — PASS (31); full suite
  **395 passed, 0 failed**.
- Đo workload thực tế (2500×20, RF, 5-fold CV): **GA 102s → 33s (3.1×);
  BO 78s → 22s (3.6×)**. Workload đồ chơi (<0.2s/fit): overhead dispatch
  Windows nuốt lợi ích — kỳ vọng đúng.

### Rủi ro còn lại

- BO batch làm giảm nhẹ sample-efficiency so với tuần tự (trade-off chuẩn
  của constant liar) — batch_size=1 để về hành vi cũ.
- Dimension inference cho phép BO đề xuất giá trị NGOÀI list gốc (trong
  range) — đúng mong muốn nhưng khác semantics grid; tắt bằng
  infer_dimensions=false nếu cần đúng list.

## 2026-07-27 — HPO-BENCH-001 + HPO-FIX-002 (benchmark dữ liệu thật)

### Nguồn dữ liệu

- OpenML API **504 sau redirect** (server họ sập, không phải mạng mình) →
  `automl/search/datasets_real.py` dùng 6 dataset thật OFFLINE: iris, wine,
  breast_cancer, digits (sklearn bundled) + glass, online_shoppers (CSV có
  sẵn trong repo). 150→12.330 hàng, 4→64 feature, 2→10 lớp.
- `scripts/benchmark_hpo.py`: grid vét cạn 18 tổ hợp làm mốc; 4 strategy còn
  lại cùng budget 8; đo cv score, wall-clock, số đánh giá, và **holdout test**.
  Chạy tuần tự độc chiếm CPU (timing mới hợp lệ).

### 2 defect benchmark lộ ra — đã vá

1. **GA phình budget**: không gian categorical ≤100 tổ hợp thì GA tự ép
   coverage 2× toàn grid, **bỏ qua budget người dùng** (đặt 4×2=8 → chạy
   18×3=**54** đánh giá, gấp 3 lần grid vét cạn). Vá: budget đặt tường minh
   là bất khả xâm phạm; auto-adjust có **trần = đúng số tổ hợp** cả hai chiều
   (mặc định 10×5=50 cho không gian 9 tổ hợp cũng bị cắt).
2. **BO tạo pool joblib lệch kích thước**: `Parallel(n_jobs=batch_size=8)`
   khác `n_jobs=-1` (16) của các strategy khác → loky giữ **2 pool = 24 tiến
   trình trên 16 lõi**; một lần chạy iris **stall 2231s**. Vá: dùng đúng
   `n_jobs` của config để tái dùng pool sẵn có.

### Kết quả đo (6 dataset, budget 8, cv=3, n_jobs=-1)

| strategy | trước | sau | nhanh hơn | evals | cv | test |
|---|---|---|---|---|---|---|
| grid_search | 22.7s | 20.1s | — (mốc) | 18 | 0.9113 | 0.9366 |
| bayesian_search | 2263.3s | **12.7s** | **178.7×** | 8 | **0.9123** | **0.9376** |
| genetic_algorithm | 122.8s | **7.1s** | **17.3×** | 54→**8** | 0.9069 | 0.9182 |
| random_search | 11.0s | 10.1s | 1.1× | 8 | 0.9096 | 0.9174 |
| successive_halving | 15.1s | 12.0s | 1.3× | 13 | 0.9087 | 0.9178 |

- BO loại bỏ outlier iris vẫn nhanh **2.8×** (31.7s→11.4s) — bản vá pool có
  tác dụng rộng, không chỉ ca bệnh lý.
- **BO giờ vừa nhanh hơn grid 1.7× vừa cho test score CAO NHẤT (0.9376 >
  0.9366 của grid vét cạn) dù chỉ dùng 8/18 lần đánh giá.**
- GA chất lượng giảm nhẹ vì lần đầu tiên nó thật sự chạy đúng budget 8 (trước
  đây "thắng" nhờ lén dùng 54 đánh giá — so sánh cũ không công bằng).
- Cảnh báo cho bài báo: RF khá trơ với lưới siêu tham số này nên chênh lệch
  chất lượng nhỏ; riêng `glass` (214 hàng, 6 lớp) tách biệt rõ:
  grid/BO test 0.8372 vs random/SH 0.7209.

### Verification

- `pytest tests/test_datasets_real.py` — PASS (15);
  `tests/test_search_strategies.py` — PASS (34, gồm 3 guard hồi quy cho 2 bug
  trên); full suite **413 passed, 0 failed**.
- Artifact: `benchmarks/hpo_real_before.json`, `hpo_real_after.json` + log.

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
