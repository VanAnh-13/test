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

## 2026-07-27 — HPO-LARGE-001 ⚠️ ĐÍNH CHÍNH SỐ LIỆU + benchmark quy mô lớn

### ⚠️ Số liệu HPO-BENCH-001 / HPO-FIX-002 KHÔNG HỢP LỆ — đã thay thế

Audit đối kháng 31 agent xác nhận **28 lỗi non-minor**, hai lỗi phá hỏng
tuyên bố chính:

1. **BO không tìm cùng không gian.** `infer_dimensions` (chính tính năng
   "cải thiện" của HPO-IMPROVE-001) biến `[50,100,200]` thành
   `Integer(50,200)` → không gian **17.667 điểm vs 18** = gấp 981 lần. Bằng
   chứng: BO trả `n_estimators=97, max_depth=12` — không có trong lưới. Phản
   chứng của auditor: ép cùng 18 điểm thì BO được 0.9330, **THUA** grid
   0.9366. Tuyên bố "BO thắng grid vét cạn" là artifact, không phải hiệu quả
   lấy mẫu.
2. **Nhánh "Bayesian" không hề Bayesian.** batch_size=8=n_calls → `ask()` đề
   xuất cả budget trước khi quan sát bất kỳ kết quả nào: **0 vòng thích ứng**,
   8 điểm giống hệt nhau trên mọi dataset.

Ngoài ra: 178.7x có 98.6% từ một outlier không tái hiện; `mean_speedup` dùng
trung bình cộng của tỉ số (thiên vị lên); grid luôn chạy đầu nên gánh warm-up;
GA "8 evals" chỉ 5 cấu hình phân biệt; SH không công bố fidelity.

### 5 lỗi code đã vá (bản vá HPO-FIX-002 trước đó CHƯA ĐỦ)

- GA: khối `adaptive_population` vẫn cắt budget khi user yêu cầu NHIỀU hơn →
  bỏ qua khi `population_size` đặt tường minh.
- GA: `self.config['generation'] = new_gen` rò rỉ budget thu nhỏ sang model
  sau → dùng biến cục bộ `generations`, truyền vào `_create_next_generation`.
- GA: trần "cả hai chiều" vẫn nới lên với không gian 42–100 tổ hợp (tới 2.4×)
  → trần CHỈ GIẢM.
- BO: `batch ≤ ceil(n_calls / min_adaptive_rounds)` (mặc định 4 vòng).
- BO: lịch sử early-stop ghi MỘT mốc mỗi batch (trước ghi từng điểm → cắt
  sớm ngay trong batch đầu).

### 5 lỗi phương pháp đã vá trong benchmark

`infer_dimensions=False` (cùng không gian), warm-up theo từng dataset,
speedup = tỉ số TỔNG (không phải trung bình tỉ số), `--n-seeds` có sai số,
công bố `n_distinct_configs` + `full_fidelity_budget` + `n_off_grid_configs`,
dọn loky pool giữa các lần đo, cảnh báo khi CPU bận.

### KẾT QUẢ TRUNG THỰC

**6 dataset nhỏ (3 seed):** grid tốt nhất chất lượng (test 0.9315±0.068);
random 1.64× là thứ duy nhất nhanh hơn; BO **0.57×** và GA **0.86×** CHẬM hơn
grid; SH 1.05×. Sai số ±0.07–0.08 **lớn hơn mọi khoảng cách giữa các thuật
toán** → khác biệt chất lượng không có ý nghĩa thống kê ở quy mô này.

**Covertype 581.012×54, 251MB (máy sạch: 0 python proc, CPU 9%):**

| strategy | test | evals | full-fid | giây | ×grid |
|---|---|---|---|---|---|
| successive_halving | 0.8413 | 13 | **3.0** | **238.3** | **3.62×** |
| genetic_algorithm | 0.8418 | 8 (5 distinct) | 8.0 | 313.9 | 2.75× |
| random_search | 0.8413 | 8 | 8.0 | 603.0 | 1.43× |
| grid_search | 0.8413 | 18 | 18.0 | 861.9 | 1.00× |
| bayesian_search | 0.8413 | 8 | 8.0 | 1086.1 | 0.79× |

**Phát hiện chính cho bài báo: thứ hạng ĐẢO CHIỀU theo quy mô.** SH từ 1.05×
(vô dụng, dữ liệu nhỏ) thành **3.62× (nhanh nhất)** ở 251MB — vì ngân sách
quy đổi full-fidelity chỉ 3.0 so với 18 của grid, mà độ chính xác y hệt. Đúng
lý thuyết multi-fidelity: chỉ lãi khi một fit đủ đắt (đo được: 15.6s full vs
2.1s trên 1/9). BO chậm hơn grid ở CẢ HAI quy mô — với 8 đánh giá, chi phí
fit GP không bao giờ khấu hao được.

### Hạn chế nêu rõ

- Covertype **n=1 seed** (~50 phút/lần chạy): cột ±0.0000 nghĩa là KHÔNG có
  phương sai để báo cáo, KHÔNG phải phương sai bằng 0.
- Một model (RandomForest), một lưới 18 tổ hợp, cv=3.
- Bản chạy nhiễu (CPU 52%) vs sạch chênh <15% và giữ nguyên thứ hạng → thứ
  hạng ổn định, con số tuyệt đối thì không.

### Verification

- `pytest tests/test_search_strategies.py` — PASS (39, gồm 6 guard hồi quy);
  `tests/test_datasets_real.py` — PASS (16).
- Artifact: `benchmarks/hpo_real_fair.json`, `hpo_large_clean.json`
  (+ `hpo_large.json` bản nhiễu để đối chiếu).

## 2026-07-27 — AGENT-T1-001 (bắt đầu giai đoạn "WM vào agent thật")

### Phạm vi

- Vá NameError trong `stream_agent` (graph.py: dùng `get_agent_registry`
  không import — sập ngay lần streaming đầu tiên).
- `get_default_model_config` giờ RAISE kèm danh sách tên hợp lệ khi
  `default_model`/`LLM_DEFAULT_MODEL` không khớp — chặn hai thảm họa: hóa
  đơn API bất ngờ và thí nghiệm chạy nhầm model không ai biết.
- Strict resolve lộ ra `ci-mock` (conftest + CI dùng làm default) CHƯA TỪNG
  được đăng ký — sống nhờ chính fallback âm thầm. Đã đăng ký `ci-mock` thành
  entry `openai_compatible` thật trỏ mock server 11435 trong hagent.yaml.

### Verification

- `pytest tests/test_llm_config_strict.py` — PASS (4); full suite
  **423 passed, 0 failed**.

### Handoff

- Kế tiếp T2: nối usage tracker vào create_chat_model + cost_metrics.

## 2026-07-27 — AGENT-T2-001

### Phạm vi

- `usage_tracker.py`: set/get/reset current tracker qua contextvars — mỗi
  run một tracker, an toàn song song (task copy context lúc tạo).
- `create_chat_model(callbacks=...)`: không truyền → tự nhặt tracker từ
  contextvar; callbacks vào CONSTRUCTOR cả 4 provider (giữ class →
  bind_tools của coordinator/subagents không vỡ); explicit thắng.
- `run_agent`/`stream_agent`: tạo tracker từ `llm.usage_tracking`, set
  contextvar, merge `tracker.summary()` vào `cost_metrics` (tokens + USD giờ
  xuất hiện trong response API lẫn JSON của run_agent_training).
- Subagents/coordinator KHÔNG sửa — nhặt tự động.

### Verification

- 19 passed (wiring + tracker); full suite **430 passed, 0 failed**.

### Handoff

- Kế tiếp T3: scripts/train_outcome_model.py + memoize _default_outcome_model.

## 2026-07-27 — AGENT-T3-001

### Phạm vi

- `scripts/train_outcome_model.py`: train head + ensemble từ Mongo
  world_trajectories hoặc --trajectories-jsonl (artifact versionable), config
  đọc từ world_model.* trong hagent.yaml (một nguồn sự thật cho vocab), ghi
  checkpoint đúng path yaml, in SHA256 + NLL; --dry-run; fail rõ khi thiếu
  sample. Đây chính là bước "nối WM vào agent thật" — đường auto của
  builder/runner đã trỏ sẵn vào các path này.
- `wm_hooks._default_outcome_model`: memoize theo mtime checkpoint (trước
  đây dựng lại ensemble từ đĩa MỖI lần một variant hoàn thành).

### Verification

- 7 passed (test mới, không cần Mongo); full suite **437 passed**.
- Test then chốt: checkpoint script sinh ra nạp được qua chính
  `_default_outcome_model` — đúng đường production.

### Handoff

- Kế tiếp T4: bật random_search + successive_halving cho campaign, version
  checkpoint v2 (đổi vocab = đổi chiều feature).

## 2026-07-27 — AGENT-T4-001

### Phạm vi

- Campaign dùng được đủ 5 thuật toán HPO (thêm random_search +
  successive_halving): pool `agent.campaign.search_algorithms`, vocab
  outcome_head/ensemble (v2), validator (đọc từ campaign config — không
  drift), enum skills tools.yaml, fallback cứng builder.
- Bump checkpoint path v2 (`outcome_head_v2.npz`, `outcome_ensemble_v2/`)
  vì đổi vocab = đổi chiều feature — checkpoint v1 không bị nạp nhầm.
- Phát hiện + vá: `outcome_ensemble` THIẾU key search_algorithms → member
  lệch vocab với head; test khóa head==ensemble.

### Verification

- 7 passed (test mới); full suite **444 passed**.

### Lưu ý vận hành

- Từ giờ vocab v2 ĐÓNG BĂNG tới khi nộp bài. Checkpoint train bằng
  `scripts/train_outcome_model.py` với config hiện hành.

### Handoff

- Kế tiếp T5: surfacing outcome surprise + vòng mở rộng campaign.

## 2026-07-27 — AGENT-T5-001 (cơ chế chính của bài báo trên agent thật)

### Phạm vi

- Vòng mở rộng campaign theo outcome surprise: tại all-terminal→comparing,
  variant completed nào có zscore HIGH (tính LẠI tại chỗ — event xuyên tick
  không tin được, model đã memoize nên rẻ) → `propose_extension_variants`
  (planner exploration_weight boost 0.5; fallback thuật toán chưa thử) →
  quay lại submitting, event `campaign_extended`. Gate
  `agent.campaign.surprise_extension` mặc định TẮT (bật cho điều kiện C).
- `Campaign.extension_rounds` vào schema (sống qua to_dict/from_dict).
- Vá bug promote sai key ở nodes + hierarchy_node: event outcome (key
  "outcome") giờ vào state["surprise"] qua `_select_surprise` (ưu tiên
  outcome zscore cao nhất, fallback latent như cũ).

### Verification

- 11 passed (E2E mở rộng → hoàn tất, quota, gate, roundtrip); full suite
  **455 passed, 0 failed**.

### Handoff

- Kế tiếp T6: harness scenario wm_campaign (guard hồi quy chuỗi event).

## 2026-07-27 — AGENT-T6-001 (khép cụm "WM vào agent thật": T1–T6 DONE)

### Phạm vi

- Scenario `wm_campaign.yaml` (tag wm_ext): expect chuỗi event
  campaign_outcome_surprise → campaign_extended, chạy qua run_graph_scenario
  (hierarchy leaf = đường production). Ghi chú điều kiện live (checkpoint +
  gate) ngay trong YAML.
- Test guard 2 chiều: có gate+model → pass; KHÔNG gate → scenario phải FAIL
  (guard không pass rỗng). Fixture checkpoint không commit nhị phân — head
  train tại chỗ <1s trong test.

### Verification

- 3 passed; full suite **458 passed**; harness smoke 16/16 OK.

### Trạng thái giai đoạn (kế hoạch 12 task)

- DONE: T1 bugfix prod, T2 usage tracker, T3 checkpoint pipeline, T4 5 thuật
  toán + vocab v2, T5 vòng mở rộng surprise, T6 harness guard.
- CÒN LẠI: T7 MPC budget, T8 goal_parser, T9 Ollama (cần user cài),
  T10 runner ma trận, T11 khung LaTeX, T12 per-request model.

## 2026-07-27 — AGENT-T7/T8/T12 (8/12 task giai đoạn agent DONE)

### T7 — MPC budget

- Campaign.total_budget (constraints max_jobs, fallback 2n) + spent_budget
  (tăng mỗi submit), sống qua dict roundtrip; vòng mở rộng tôn trọng budget
  (cạn → không mở rộng; n_extra cap theo remaining); backend cem_mpc_v1 →
  plan_batch với remaining/total (spy test xác nhận n=2/remaining=5/total=8).

### T8 — goal_parser thuật toán

- 5 thuật toán bắt được từ EN+VN; phát hiện kèm: constraints.search_algorithm
  trước đây bị diversify GHI ĐÈ (yêu cầu user vô tác dụng) → ghim variant
  đầu source='requested'.

### T12 — per-request model

- run_agent/stream_agent(model_name) validate strict TRƯỚC khi chạy graph;
  contextvar per-run cho coordinator/subagents; req.model truyền thật ở cả
  3 call site chat_router, tên sai → HTTP 400 kèm danh sách hợp lệ.

### Verification

- 8+21+8 test mới pass; full suite **495 passed, 0 failed**.

### Còn lại

- T9 Ollama (CHỜ USER CÀI), T10 runner ma trận, T11 khung LaTeX,
  DEPLOY-DOCKER (user yêu cầu thêm — docker-compose*.yaml protected, yêu cầu
  trực tiếp của user = phê duyệt).

## 2026-07-27 — T10/T11/DEPLOY-001 (11/12 task + deploy DONE)

### T10 — Runner ma trận thí nghiệm

- `run_experiment_matrix.py`: RealJobEnv chạy sklearn THẬT in-process cho
  mỗi start_training (LLM thật + job thật, không docker, không mock score);
  điều kiện A/B/C/C_mpc = yaml overlay tạm (A trỏ checkpoint __disabled__);
  JSONL resumable (ô lỗi chạy lại), mỗi dòng đủ route/campaign/events/
  tokens/USD/SHA. Dry-run 54 ô đúng.

### T11 — Khung bài báo

- `paper/main.tex` (article chuẩn, swap style ACML khi công bố ~7/8) +
  `references.bib` 13 entry + `make_paper_tables.py` (bảng sinh TỰ ĐỘNG từ
  JSONL/JSON thật — test xác nhận 3.62× covtype vào đúng bảng).

### DEPLOY-001 — Docker deploy (user yêu cầu; compose protected → user
### approve trực tiếp)

- Healthcheck toolkit(/home)/bridge(chat/health)/mongo(ping)/minio(mc ready);
  bridge chờ toolkit service_healthy; toolkit chờ mongo healthy.
- `deploy.llm.env.example` (trước RỖNG) đủ mọi knob + cảnh báo strict
  resolve; `docs/deploy.md` 6 bước gồm train checkpoint + smoke
  --require-live. `docker compose config` sạch.

### Verification tổng

- Full suite **510 passed, 0 failed** (baseline nhận repo: 207).
- Trạng thái 12 task: DONE T1–T8, T10–T12 + DEPLOY. CÒN: T9 Ollama (chờ
  user cài) → hiệu chuẩn latency → chạy ma trận thật (checkpoint trước).

## 2026-07-27 — LLM-META-001 (endpoint Meta AI của user)

### Phạm vi
- Đăng ký endpoint OpenAI-compatible user cung cấp: `https://api.meta.ai/v1`,
  model duy nhất `muse-spark-1.1` (xác nhận qua GET /v1/models → 200).

### Quyết định
- Key CHỈ nằm trong `src/backend/.env` (gitignored — xác minh
  `git check-ignore`); `hagent.yaml` tham chiếu `${META_AI_API_KEY}`,
  không bao giờ literal (repo public). `.env` protected — user cung cấp
  key trực tiếp để cấu hình = phê duyệt ghi vào task.
- Không thêm pricing muse-spark (không có bảng giá công khai) → cost USD = 0
  theo thiết kế, tokens vẫn đếm.

### File thay đổi
- `hagent.yaml` (+ model `meta-ai`), `deploy.llm.env.example` (+ block
  META_AI_*), `.env` (local, KHÔNG commit), feature_list.json.

### Verification
- `require_model_config('meta-ai')` → openai_compatible/muse-spark-1.1 — PASS
- Smoke live qua `create_chat_model('meta-ai')`: "15+27?" → "15 + 27 = 42";
  usage tracker in=23 out=390 calls=1 — PASS
- `git grep <key>` tracked files → exit 1 (key không có trong repo) — PASS
- Full suite `pytest -m "not ollama"` → **510 passed** — PASS

### Handoff
- Dùng được ngay: `LLM_DEFAULT_MODEL=meta-ai`, per-request `{"model":
  "meta-ai"}`, hoặc thêm vào `models:` trong agent_matrix_config.yaml.
- CÒN: T9 Ollama (user cài) → checkpoint train → ma trận.

## 2026-07-27 — DOCKER-BACKEND-002 + OPENCLAW-REMOVE-001 + DEERFLOW-BRAND-001

### Phạm vi (3 yêu cầu user liên tiếp, verify gộp một lần)
1. "viết lại docker phía backend"; 2. "gỡ toàn bộ mọi thứ liên quan đến
openclaw"; 3. "Không nói là DeerFlow, mà chỉ dựa trên công nghệ của họ".

### Quyết định
- MỘT image backend `hautoml-toolkit` (python:3.12-slim + libgl/libgomp)
  dùng chung toolkit/worker/nano — XÓA worker.dockerfile + nano.dockerfile;
  worker/nano có build section trong compose (trước không tự build được).
- `.dockerignore` viết lại — CHẶN `.env*` (lỗ bảo mật: COPY . . từng có
  thể bake API key vào image); loại .venv/tests/paper/kết quả benchmark.
- Bridge: +numpy (predictor lazy-import), bỏ hack echo __init__.py.
- OpenClaw gỡ tận gốc: xóa proxy.py + SOUL.md + skills/ (không consumer
  python nào đọc — đã grep); compose bỏ service openclaw_gateway + 3
  volume + HAGENT_RUNTIME_MODE/HAGENT_GATEWAY_URL/HAGENT_HOOKS_TOKEN;
  bridge bỏ _call_openclaw_gateway + get_gateway_config/get_hooks_config;
  hagent.yaml bỏ section gateway/hooks/skills/proxy. error_messages
  chuyển về section top-level (bắt được bug key trùng yaml khi di chuyển).
- Branding: provider = "hagent" (graph/bridge/router), env
  HAGENT_DEERFLOW_URL → HAGENT_AGENT_RUN_URL, bỏ DEERFLOW_BASE_URL,
  _call_deerflow_runtime → _call_agent_runtime, docstring "HAgent — ...".
  GIỮ attribution: "inspired by DeerFlow 2.0 (ByteDance)" + dòng
  Reference/Tham chiếu (~10 dòng) — đúng yêu cầu "dựa trên công nghệ".
- Protected files đụng tới theo yêu cầu trực tiếp của user: compose,
  app.py (comment-only), requirements.txt (1 dòng comment),
  bridge/requirements.txt (+numpy).

### Sự cố tự bắt được
- Chẩn đoán ban đầu "3 dockerfile mất" SAI (glob Dockerfile* không khớp
  tên thường) — sửa lại thành viết lại thật, đã báo user.
- sed xóa dòng ENV cuối để lại backslash treo → ENV nuốt RUN trong
  toolkit dockerfile — bắt ngay khi soát diff, sửa trước khi build.

### Verification
- grep openclaw = 0 hit; grep brand tokens = 0 hit — PASS
- pytest -m "not ollama" → **514 passed** — PASS
- compose config 3 profiles exit 0; build 2 image exit 0 — PASS
- Smoke up THẬT: mongo/kafka/minio/toolkit/bridge CẢ 5 healthy;
  toolkit /home 200; bridge health connected:true, provider "hagent";
  down sạch — PASS

### Handoff
- MATRIX-META-001 tiếp tục: warmup 120/120 XONG → train checkpoint v2 →
  hiệu chuẩn meta-ai → pre-flight audit (workflow resume wf_79bde535-bc7)
  → full 54 ô.

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

## 2026-07-27 - PLAN IMPLEMENTATION START

### Scope decision

- User explicitly approved implementation of the multi-agent continuation plan.
- `MATRIX-META-001` is blocked at adversarial pre-flight: the existing row has zero LLM calls, ensemble members were not exact-whitelisted, and table generation was not fail-closed.
- WIP moved to prerequisite `WM-ARTIFACT-002`; exact head/member/manifest/trainer/test paths are now whitelisted.

### Working-tree baseline

- Fixed review point: `b63f0fb9031461a327930232038073d7d3338a3a`.
- Preserved pre-existing untracked artifacts: matrix results, outcome head v2, and five ensemble members.
- Canonical Windows harness passed with Git Bash and `PYTHONUTF8=1`.

### Handoff

- Maker implements an atomic, verifiable checkpoint manifest using test-first slices.

### Scope extension approval

- User replied `continue` after the exact atomic-write blocker was reported.
- Added exact transient path `src/backend/data/world_model/outcome_ensemble_v2/manifest.json.tmp`; it may exist only during same-directory fsync + os.replace and must be absent after success or failure.
- Root took over as the sole Maker after the delegated Maker exhausted its usage allocation.

## 2026-07-27 - WM-ARTIFACT-002 BLOCKED ON FULL-SUITE ISOLATION

### Scope and implementation

- Added atomic same-directory manifest publication with fsync, os.replace, stale-manifest invalidation, and cleanup after write or self-validation failure.
- Manifest binds the warmup source, effective model config, vocabulary, full training schema, head, and exact ensemble member set with full 64-character SHA-256 values.
- Effective config records the executed ensemble size and omits storage paths; validator cross-checks config and vocabulary against non-pickled metadata in every NPZ.
- Canonical NPZ archives were normalized so validation can use allow_pickle=False.
- Public validation rejects missing, extra, tampered, misdeclared, incomplete, or semantically mismatched artifacts.

### Canonical artifact evidence

- Warmup: 120 trajectory documents and 120 extracted samples.
- Source SHA-256: `7098cc0b8f2cceb0249a52523e65fb0c84870f256712402d81d32878e42e239f`.
- Config SHA-256: `6dd87a982eeebc57bdbd680c0793aabe2aeccf3c35081552f3c05025a12dafc2`.
- Vocabulary SHA-256: `8711d6f8fda46cde285a35adc04fba8527d8e98eb6d651efc1dfb50f0de358e0`.
- Head SHA-256: `9be6d5233ac7cc77c2f1d3251f5d5887b756ccb967610ea60296e30afe38f07e`.
- Member SHA-256 values: `9be6d5233ac7cc77c2f1d3251f5d5887b756ccb967610ea60296e30afe38f07e`, `2532a502b13f6896e3f09beb312bc12f4e270fa573c79281e5e80231722105ca`, `a44f5381fe704530c3bd45e3d8c814cf3bea3b34a7513da79c4303fa0136158f`, `044894c552a1dd4fee0735fcaaf2952006ab2063174eb3e03848a554eee88af1`, `5e70751349c912ccd05e16c6c1e0a1d91a5ac8bcd48f0929584a0e1a95531e11`.
- Public validator passed with expected_k=5; exactly five members exist; source hash matches; no manifest temp remains; every NPZ metadata array loads with allow_pickle=False.

### Verification

- Targeted pytest: `27 passed`.
- Trainer dry-run: exit 0, 120/120.
- Canonical trainer: exit 0, head plus five members plus manifest regenerated.
- Independent Standards Checker: no implementation blocker.
- Independent Spec Checker: all WM-ARTIFACT-002 acceptance criteria met.
- Ruff comparison: current task files add no lint findings relative to fixed point `3369d28` (10 existing trainer findings and 3 existing test findings remain unchanged).
- Full non-Ollama suite: `533 passed, 1 failed, 7 deselected`.

### Blocker and handoff

- The only failure is `tests/test_cem_config_planner.py::TestBuilderIntegration::test_builder_without_model_unchanged`.
- That test claims a no-model scenario but omits `outcome_model=None`; its default `auto` now loads the required canonical checkpoint and correctly emits a `wm_planner` variant.
- Minimal repair: explicitly pass `outcome_model=None` in that test, then rerun the isolated test and full suite.
- Exact path `src/backend/tests/test_cem_config_planner.py` is outside WM-ARTIFACT-002 whitelist and has not been modified. User approval is required before adding it.

## 2026-07-27 - WM-ARTIFACT-002 RESUMED

- User replied `ok`, explicitly approving `src/backend/tests/test_cem_config_planner.py`.
- WIP reopened for the same task; the only intended code change is to pass `outcome_model=None` in the no-model integration test.
- Public TDD seam: `build_campaign(..., outcome_model=None)` must preserve the no-world-model campaign behavior even when canonical checkpoints exist on disk.

## 2026-07-27 - WM-ARTIFACT-002 DONE

- User-approved isolation fix: `test_builder_without_model_unchanged` now passes `outcome_model=None`; no production code changed in the unblock slice.
- RED evidence before approval: isolated test failed because canonical checkpoints were loaded through the default `auto` seam.
- GREEN evidence: isolated test 1 passed; complete planner test file 22 passed; full non-Ollama suite 534 passed and 7 deselected.
- Final Standards and Spec Checkers found no blocker, scope creep, or file outside the approved whitelist.
- WM-ARTIFACT-002 meets Definition of Done; WIP released.

## 2026-07-27 - MATRIX-PROTOCOL-001 START

- Orchestrator advanced to prerequisite task two after WM-ARTIFACT-002 passed.
- Scope is limited to the matrix runner, its public tests, two protocol sidecars, the existing result JSONL, and control files.
- No live Meta call or paid experiment is authorized in this task; tests must use a boundary fake and dry-run must remain call-free.
- Scout performs read-only dependency and adversarial protocol review before the sole Maker edits production code.

## 2026-07-27 - MATRIX-PROTOCOL-001 BLOCKED

- Both read-only Scouts confirmed the existing zero-call row must be rejected and removed before rerun.
- Crash-safe migration needs the exact sibling temporary path `src/backend/benchmarks/agent_matrix_results.jsonl.tmp`, which is not yet in the approved whitelist.
- No production or test file was edited and no Meta/network call was made.

## 2026-07-27 - MATRIX-PROTOCOL-001 RESUMED

- User approved the exact additional whitelist path `src/backend/benchmarks/agent_matrix_results.jsonl.tmp` and requested implementation to continue.
- WIP returned to MATRIX-PROTOCOL-001; root remains the sole Maker.
- Public seams are design/payload, injected advice boundary, strict sidecar/resume migration, evidence-bound cell execution, and call-free dry-run CLI.

## 2026-07-27 - MATRIX-PROTOCOL-001 DONE

### Scope and decisions

- Froze the main matrix at A/B/C x six datasets x one model x three seeds (54 cells); C_mpc remains outside this protocol.
- Bound the design SHA to dimensions, condition patches, prompt/job config, the seven-field anonymous meta-feature schema, the fixed five-algorithm pool, and zero provider retries.
- Added paired advice journaling with pending/dispatched/accepted states. Advice is deduplicated once per dataset/model key and reused across A/B/C and all seeds.
- Fail-closed behavior rejects malformed/duplicate-key/non-finite JSON, invalid enum, network failure, zero usage, conflicting execution evidence, dataset/prompt/design drift, and incomplete resume rows.
- A dispatched claim is never retried automatically. Provider-authenticated reconciliation is not available, so ambiguous dispatched claims remain blocked rather than being accepted from an unverified receipt.
- No live or paid Meta request was made; all provider behavior was verified through an injected no-network boundary.

### Evidence and artifacts

- Frozen design SHA-256: `0860d3662ed8a2420aa46887cce84fdb70787c34f5b2271690976905bb4893bb`.
- Legacy `A:iris:meta-ai:0` had zero calls and incomplete protocol evidence. It was migrated idempotently to `agent_matrix_preflight_rejected.jsonl`; no result row was retained.
- Rejected sidecar SHA-256: `63546511053c3738834f26902120c52ab2c89d2a4abb47fd34d839b7957abe83`.
- Original canonical row SHA-256 inside the rejection record: `de45ce9a052a08dae1451538e720ee0fe6d5559e4fb98197793b1c7e5f121127`.
- Rejection ID: `af426b42df6fc77c9070cfba93ea82cb0bd41e5b066a567e1cddcea8ddfc7b57`.
- Strict validation confirmed one rejection, `cell_usage_invalid`, no raw prompt/response/secret fields, 64-hex hashes, and byte-identical second migration.
- Final dry-run reports 54 todo cells; results, temp, and advice sidecars are absent, and the rejected sidecar is unchanged.

### Files changed

- `src/backend/scripts/run_experiment_matrix.py`
- `src/backend/tests/test_experiment_matrix.py`
- `src/backend/benchmarks/agent_matrix_preflight_rejected.jsonl`
- `feature_list.json`
- `claude-progress.md`

### Verification

- Targeted matrix suite: `66 passed`.
- Minimal HAGENT_CONFIG leak repro: RED `1 failed, 1 passed`; after the fix GREEN `2 passed`.
- Full suite first exposed the process-global config leak: `508 passed, 82 failed, 7 deselected`.
- Final full suite: `590 passed, 7 deselected` in 56.43 seconds.
- Two independent protocol Checkers passed the frozen protocol. The final regression Checker passed the environment/cache restoration fix with no blocker.
- `git diff --check` has no content error; changed files remain inside the exact task whitelist.

### Residual risk and handoff

- The code cannot guarantee distributed exactly-once recovery after a provider accepted a request but the process crashed before journaling the response. It deliberately fails closed at `dispatched`; human/provider-authenticated reconciliation is required.
- Matrix execution is still 0/54 and requires the planned live A+B go/no-go before any paid full run.
- Next prerequisite: create and execute `PAPER-TABLE-GATE-001` with an exact whitelist before reopening `MATRIX-META-001`.

## 2026-07-27 - PAPER-TABLE-GATE-001 START

- Orchestrator advanced to the third prerequisite after MATRIX-PROTOCOL-001 passed and released WIP.
- Exact write scope is limited to the table generator, its tests, two generated TeX tables, and control files.
- The gate must reject missing, malformed, duplicate, errored, wrong-design, incomplete-seed, and non-54-cell evidence before touching either output.
- Current matrix execution remains 0/54; no production table or paper claim may be generated from the rejected legacy row.
- Scout begins read-only dependency and failure-mode review before the sole Maker edits generator code.

## 2026-07-27 - PAPER-TABLE-GATE-001 DONE

### Scope and decisions

- `make_paper_tables.py` is now fail-closed: it loads matrix rows, paired-advice sidecar, frozen matrix config, WM manifest, and both HPO JSON artifacts before touching either TeX output.
- Matrix publication requires exactly 54 successful cells for A/B/C x six datasets x `meta-ai` x seeds 0/1/2, the frozen design SHA, accepted paired advice, current dataset/advice bindings, no row errors, complete traces, and stable git provenance.
- Condition A is explicitly required to have no checkpoint and no `wm_planner`; B/C must carry `wm_planner` evidence and a checkpoint SHA equal to the WM manifest head SHA.
- HPO table generation now validates raw Cartesian grids, summaries recomputed from raw rows, and the frozen budget/cv/param_grid before rendering.
- Both LaTeX outputs are rendered in memory first and written through rollback logic, so validation failure or a later write failure leaves existing table bytes unchanged.

### Files changed

- `src/backend/scripts/make_paper_tables.py`
- `src/backend/tests/test_make_paper_tables.py`
- `feature_list.json`
- `claude-progress.md`

### Verification

- `py_compile scripts/make_paper_tables.py` passed.
- `ruff format` reformatted the two task files; `ruff check` passed.
- Targeted table suite: `31 passed` with the exact task timeout command.
- Default production invocation returned exit `2` and created no `paper/tables/agent_matrix.tex` or `paper/tables/hpo_two_scales.tex` because the real matrix evidence is still absent.
- Full backend suite: `616 passed, 7 deselected`.
- Independent legacy Checker agents could not complete because old agent model configuration was unsupported or timed out; root performed a final acceptance/security/whitelist review after tests and found no blocker.

### Residual risk and handoff

- Matrix execution is still 0/54, so no production paper table exists yet. Reopen `MATRIX-META-001` only after regenerating checkpoint/manifest, running the live A+B go/no-go, freezing design/config/code, and executing the 54-cell operator run.

## 2026-07-28 - CI-BASELINE-001 START

### Scope

- User approved the sequential CI, memory, API, multi-turn, SSE/UI, E2E, and manuscript implementation plan.
- WIP opened only for `CI-BASELINE-001`; `MATRIX-META-001` remains blocked and no live or paid model call is authorized.
- Exact protected workflow scope is limited to `ci.yml`, `cd.yml`, `deerflow-automl-tests.yml`, `agent-train-student.yml`, and `student-performance.yml`.

### Baseline

- Fixed review point: `b714f7d4028821e88e9ecc451ab761bad97dd281`.
- Working tree was clean; canonical Windows harness passed with WIP=0 before activation.
- Public verification seams are workflow triggers, referenced paths, canonical pytest command, current Docker image build targets, actionlint, and an actual GitHub run on `hagent`.

### Known external gates

- `actionlint` and `gh` are not currently installed and the remote has no confirmed `hagent` workflow run.
- The task must be marked `blocked`, not `done`, if those gates cannot be executed after local implementation.

## 2026-07-28 - CI-BASELINE-001 BLOCKED

### Implemented scope

- Added `hagent` triggers for canonical CI/student workflows and for CD image validation; docs deployment remains restricted to `main` or explicit manual dispatch.
- Replaced the drifting pytest file list with the canonical full non-Ollama suite and moved the standalone mock server after pytest so the suite fixture owns port `11435` without conflict.
- CD now builds only the toolkit and HAgent Bridge Dockerfiles; worker services continue to reuse the toolkit image.
- Removed the deprecated DeerFlow workflow alias, legacy runtime settings, and stale branding inside the approved five-workflow scope.
- Tightened CD permissions to `contents: read` by default; only the docs deployment job receives `contents: write`.

### Verification

- Static retained-workflow/path checks passed; all four retained YAML files parse successfully and no approved retained workflow contains DeerFlow, OpenClaw, `worker.dockerfile`, or `HAGENT_RUNTIME_MODE` remnants.
- Full backend suite passed: `616 passed, 7 deselected in 71.30s`.
- `docker compose config --quiet` exited `0`; it emitted only expected warnings for locally unset secret variables without printing secret values.
- `docker compose build toolkit hagent_bridge` exited `0`; both images built successfully.
- `actionlint .github/workflows/*.yml` exited `1` because `actionlint` is not installed.
- `gh run list --branch hagent --limit 10` exited `1` because `gh` is not installed; no real green workflow run matching the implementation commit was verified.

### Checker review

- Spec Checker found no blocker and confirmed the mock/test ordering, current Docker paths, branch triggers, alias deletion, and whitelist scope.
- Standards Checker found one excessive-permissions blocker in CD. The default was reduced to read-only, the docs job alone was granted write access, and the Checker re-review found no remaining blocker.

### Blocker and handoff

- Local implementation is ready, but Definition of Done explicitly requires successful `actionlint` and a real green GitHub workflow run on `hagent` whose head SHA matches the implementation commit.
- Because those gates are unavailable on this machine, `CI-BASELINE-001` is `blocked`, `current_task_id` is cleared, and the next planned task may start without misrepresenting CI as complete.

## 2026-07-28 - API-CONTRACT-001 START

### Scope

- Opened exactly one WIP task after `CI-BASELINE-001` was safely blocked.
- Maker whitelist is limited to Bridge models/app, toolkit chat router, the dedicated contract test, and control files.
- No auth, root `app.py`, package/lockfile, compose, secret, paid API, or live external call is authorized.

### Baseline and public seams

- Fixed review point: `583e9963dbd9`.
- Public request seam: sync chat forwards `{message, conversation_id, context, model}`; upload additionally forwards `model`.
- Public response seam: provider/model/route plus tool, planning, campaign, hierarchy, world-model, evaluation, execution, revision, and cost metadata.
- Failure seam: invalid model `400`, upstream 4xx preserved, network `502`, timeout `504`, never fake HTTP `200` success.
- Provider discovery must reflect the toolkit registry, not a duplicated hard-coded list.

## 2026-07-28 - API-CONTRACT-001 DONE

### Implemented scope

- Standardized Bridge and toolkit request forwarding for `message`, `conversation_id`, public `context`, and `model`; upload endpoints now accept and forward `model`.
- Kept JWT exclusively in the Authorization header, removed request-header logging, stripped principal/token fields from runtime JSON, and made persisted server world state authoritative over forwarded snapshots.
- Exposed provider/model/route, tool, planning, campaign, hierarchy, world-model, evaluation, execution event/log, revision, and cost metadata through both response schemas.
- Replaced the hard-coded provider list with aliases grouped from the shared toolkit model registry.
- Made invalid models fail with HTTP 400, preserved upstream 4xx status, mapped network failures to 502 and timeouts to 504, and rejected malformed or 5xx runtime responses with 502 instead of fake HTTP 200 success.

### Verification

- Targeted contract suite: `27 passed`.
- Full backend non-Ollama suite: `643 passed, 7 deselected in 70.53s`.
- The first sandboxed full-suite attempt was discarded because Windows denied pytest temp directories and joblib named pipes; the approved outside-sandbox rerun exited 0.
- Focused Ruff `E402,F,I` check on all four task code/test files exited 0. Full Ruff comparison reduced baseline findings from 44 to 36 and introduced no new finding class.
- `git diff --check` and JSON validation exited 0; the working set contains only the six exact whitelisted paths.

### Checker and handoff

- Two independent read-only Checker launches were attempted after code freeze, but the environment stopped both at startup because the sub-agent usage quota was exhausted.
- Following the explicit `AGENTS.md` fallback, root performed a separate acceptance, trust-boundary, error-semantics, lint-delta, test-sufficiency, and whitelist review and found no blocker. This limitation is recorded rather than represented as an independent review.
- Multi-turn history remains intentionally deferred to `CHAT-MULTITURN-001`; this task only stabilizes the transport and response contract.

## 2026-07-28 - CHAT-MULTITURN-001 START

### Scope

- Opened exactly one WIP task after committing `API-CONTRACT-001` at `104fa4bd1def`.
- Maker scope is limited to Bridge app, toolkit chat router, agent graph, the dedicated multi-turn test, and control files.
- Bridge remains the sole conversation-history source; auth, persistence modules, root app, package/lockfile, compose, and SSE behavior are outside this task.

### Acceptance seams

- Load at most 20 prior owner-scoped messages and permit only `user` or `assistant` roles.
- Forward history through `context.history` without credentials or spoofable principal fields.
- Convert history to ordered human/AI graph messages and append the current user message exactly once.
- Prove owner isolation, genuine second-turn continuity, ordering, and no duplication before the full suite.

## 2026-07-28 - CHAT-MULTITURN-001 DONE

### Implemented scope

- Bridge now snapshots at most 20 persisted messages before writing the current turn, using the existing owner-scoped `(conversation_id, user_id)` query.
- History is reduced to ordered string messages with role `user` or `assistant`; client-supplied history and principal fields are not accepted by the public Bridge context.
- Both text and upload calls forward the trusted history as `context.history` to the private toolkit route.
- Toolkit sanitizes the history again, and the graph maps it to ordered `HumanMessage`/`AIMessage` objects before appending exactly one current `HumanMessage`.

### Verification

- TDD red state was observed with four expected failures before implementation; final targeted suite: `4 passed`.
- API contract regression suite: `27 passed`.
- Full backend non-Ollama suite: `647 passed, 7 deselected in 56.72s`.
- Focused Ruff `E402,F,I`, Python compile, `git diff --check`, and JSON validation all exited 0.
- Tests cover genuine second-turn continuity, same-conversation owner isolation, 20-message bounding, role/content filtering, graph message ordering, and current-message non-duplication.

### Checker and handoff

- Independent sub-agent review remained unavailable because the environment quota was exhausted immediately before this task.
- The explicit `AGENTS.md` fallback self-review checked owner scope, trust boundaries, message ordering, regression risk, and the six-file whitelist and found no blocker.

## 2026-07-28 - MEM-CORE-001 START

### Scope

- Opened exactly one WIP task after committing `CHAT-MULTITURN-001` at `90ac883abf2c`.
- Maker scope is limited to the middleware chain, fact extractor, existing phase-3 context test, and control files.
- Persistence backend changes are deferred to `MEM-MONGO-002`; no Mongo, YAML, auth, package, or API file is in scope.

### Acceptance seams

- Post middleware consumes the real `result.tool_outputs` and `result.response` values.
- `from_tools` and `from_responses` gate their sources independently.
- Error material and unconfirmed user text never become facts.
- Response keys are deterministic SHA-256 values across Python processes.

## 2026-07-28 - MEM-CORE-001 DONE

### Implemented scope

- `MemoryMiddleware.post_process` now reads the real `result.tool_outputs` and `result.response` seams instead of the absent legacy `result.messages` seam.
- Tool and response extraction are independently gated by `memory.extraction.from_tools` and `from_responses`.
- Failed tool envelopes/payloads and failed/error responses are rejected before extraction; assistant suggestions are no longer inferred as user preferences.
- Response-derived fact keys use the full deterministic SHA-256 hexadecimal digest instead of randomized Python `hash()`.

### Verification

- TDD red state was observed with six expected failures before implementation; final targeted phase-3 suite: `42 passed`.
- Full backend non-Ollama suite: `655 passed, 7 deselected in 58.13s`.
- Focused Ruff `E402,F,I`, Python compile, `git diff --check`, and JSON validation exited 0.
- Tests cover both extraction flags in all combinations, direct tool/response inputs, rejection of legacy messages and errors, SHA-256 determinism, and no assistant-suggestion preference inference.

### Checker and handoff

- Independent sub-agent review remained unavailable because the environment quota was exhausted.
- The explicit `AGENTS.md` fallback self-review checked config semantics, failure filtering, provenance/source boundaries, deterministic keys, regression risk, and the five-file whitelist and found no blocker.
- Mongo persistence remains deferred to the next task, `MEM-MONGO-002`.

## 2026-07-28 - MEM-MONGO-002 START

### Scope

- Opened exactly one WIP task after committing `MEM-CORE-001` at `960d37b4481f`.
- Maker scope is limited to the Mongo fact store, memory factory, memory YAML stanza, dedicated tests, and control files.
- No dependency, lockfile, compose, auth, API, or migration file is in scope.

### Acceptance seams

- Compound unique identity is `(user_id, key)` and every operation is owner-scoped.
- Save is an upsert while reads reconstruct the existing `Fact` interface.
- `backend: auto` chooses Mongo only when `MONGODB_CONNECT` is configured; absence uses local dev/test storage.
- Any configured Mongo initialization/runtime failure is surfaced and never converted into a local fallback.

## 2026-07-28 - MEM-MONGO-002 COMPLETE

### Delivered

- Added an asynchronous `MongoFactStore` with a unique `(user_id, key)` index, owner-scoped CRUD/search, access counting, bounded results, and upsert persistence.
- Changed memory `backend: auto` to select Mongo only from the presence of `MONGODB_CONNECT`; an empty or failing configured Mongo path now fails closed without local fallback.
- Added 11 focused tests covering storage documents, all owner boundaries, selection semantics, runtime failures, zero limits, and YAML defaults.

### Verification

- Targeted: `tests/test_memory_mongo.py` -> 11 passed.
- Regression: `tests/test_phase3_context.py` -> 42 passed.
- Full backend: `tests -m "not ollama"` -> 666 passed, 7 deselected.
- Focused Ruff `E,F,I` and `git diff --check` passed.

### Review and residual risk

- Independent sub-agent review remained unavailable because the environment quota was exhausted.
- The documented `AGENTS.md` fallback self-review checked the FactStore interface, every Mongo query for owner scoping, fail-closed selection, configured/runtime failure propagation, and the exact six-file whitelist; no blocker remained.
- No live Mongo service was required by the approved test command; production connectivity is still surfaced on the first store operation and is never converted into local persistence.

## 2026-07-28 - SSE-CORE-001 START

### Scope

- Opened exactly one WIP task after committing `MEM-MONGO-002` at `35cd831ac4d5061e0941752f4b9c1b5cbe0787a3`.
- Maker scope is limited to toolkit graph streaming, SSE encoding, the private agent-run stream route, focused tests, and control files.
- Bridge persistence, frontend behavior, dependencies, auth, compose, and package/lock files remain outside this task.

### Acceptance seams

- Frames use explicit `event`, monotonic `id`, JSON `data`, documented event names, and one terminal without `[DONE]`.
- The terminal response is reconstructed from the root final graph state, never from a concatenation of internal model-token events.
- Usage/model request context is reset in `finally`; cancellation must propagate only after cleanup.
- `/api/v1/chat/agent-run/stream` is private and stateless with respect to conversation storage.

## 2026-07-28 - SSE-CORE-001 COMPLETE

### Delivered

- Added private stateless `POST /api/v1/chat/agent-run/stream` with sanitized history, authoritative server context, explicit model forwarding, and no conversation writes.
- Replaced data-only/Sentinel streaming with typed SSE frames carrying monotonic IDs and exactly one `done` or safe `error` terminal.
- Reconstructed `done.response` from the root final graph state and preserved tool, planning, campaign, hierarchy, world-model, evaluation, execution, revision, and cost metadata.
- Reset usage/model context tokens in `finally`; cancellation closes the agent iterator and propagates without persisting a partial terminal.

### Verification

- Targeted: `tests/test_agent_streaming.py` -> 9 passed.
- Integration regression across memory, model config, usage, API contract, and multi-turn -> 84 passed.
- Full backend: `tests -m "not ollama"` -> 675 passed, 7 deselected.
- Focused Ruff checks and `git diff --check` passed; two unrelated pre-existing `chat_router.py` E501 lines remain outside this task's added lines.

### Review and residual risk

- Independent sub-agent review remained unavailable because the environment quota was exhausted.
- The documented `AGENTS.md` fallback review checked cancellation/failure/success cleanup, serialization failure, duplicate terminal suppression, secret-safe errors, owner history forwarding, middleware response extraction, and the exact six-file whitelist; no blocker remained.
- Tests use deterministic graph/runtime doubles and do not call a paid or live LLM endpoint.
