# World Model (LeWM-inspired) — Deep integration

Tham chiếu: [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312)

## Mapping

| LeWM | DeerFlow AutoML |
|---|---|
| Observation `o_t` (pixels) | `AutoMLObservation` (datasets/jobs/focus/phase/goal) |
| Action `a_t` | `AutoMLAction` (closed tool space) |
| Encoder `enc(o)→z` | `structured_v1` feature encoder |
| Predictor `pred(z,a)→ẑ` | `tabular_transition_v1` **or** `neural_jepa_v1` |
| Latent planning (CEM/MPC) | `cem_lite` planner |
| Surprise / VoE | `surprise.py` ‖ẑ − z_actual‖ |
| Offline trajectories | `TrajectoryStore` → Mongo `world_trajectories` |

## Deep integration (full stack)

```text
Frontend ChatWidget (World Model panel)
    │  plan / surprise / campaign / hierarchy chips
    ▼
Bridge (:9900) ── WorldStateStore (Mongo world_states)
    │              TrajectoryStore (Mongo world_trajectories)
    ▼
chat_router / run_agent  (_wm_service + _world_store injected)
    ▼
LangGraph: coordinator → hierarchy ⇄ campaign | plan_executor ⇄ reviser
    │              │                    │
    │              └── smart-skip + leaf WM update
    │                    surprise on train ticks
    └── encode → predict(neural|tabular) → CEM plan → surprise → trajectory
```

### Durable stores

- `WorldModelService.from_config(mongo_client=..., db_name=...)` binds trajectories.
- `run_agent` / `stream_agent` inject `_wm_service` + `_world_store`.
- Bridge persists tool outputs **and** plan/surprise/world_model meta after chat.
- Factory helpers: `world/runtime.py`, `create_trajectory_store`.

### Neural predictor (JEPA-lite)

```yaml
world_model:
  predictor:
    backend: neural_jepa_v1   # or tabular_transition_v1
    hidden_dim: 128
    checkpoint_path: ./data/world_model/jepa_v1.npz
    fallback: tabular_transition_v1
```

Offline train:

```bash
cd src/backend
python scripts/train_world_predictor.py --from-memory --out ./data/world_model/jepa_v1.npz
python scripts/train_world_predictor.py --mongo --epochs 50
```

Numpy-only MLP; no torch required. Missing checkpoint → tabular fallback.

### Agent surfaces

| Component | WM usage |
|---|---|
| Coordinator | CEM plan + encode |
| Plan executor | validate → tool → `wm.update` → surprise/revise |
| Campaign | surprise on submit/poll; jobs synced into snapshot |
| Hierarchy | leaf tool WM update; train leaf via campaign hooks |
| Middleware | load/persist snapshot when `_world_store` set |

### API / UI

`ChatResponse` optional fields: `plan_status`, `selected_plan`, `surprise`,
`cost_metrics`, `execution_events`, `world_model` (summary), `campaign_status`,
`hierarchy_status`, `evaluation`.

Frontend: collapsible **World Model** panel + per-message chips.

## CI / CD — human train prompt

Deterministic full-system path (no GPU required):

```bash
cd src/backend
# Default human prompt: train glass classification with multi-candidate campaign
python scripts/run_world_model_train_e2e.py --train-neural --json /tmp/wm_e2e.json

# Custom human prompt
python scripts/run_world_model_train_e2e.py \
  --prompt "Please train a model on dataset ds_glass. Target Type, metric f1." \
  --mongo --train-neural
```

Harness scenarios (tags `wm`):

```bash
python scripts/run_agent_harness.py --layer offline,graph --tags wm
```

| Workflow | Role |
|---|---|
| `.github/workflows/ci.yml` | Unit + harness + **World Model full-system train** step |
| `.github/workflows/world-model-train-e2e.yml` | Dedicated WM train E2E (human prompt, Mongo, neural) |
| `.github/workflows/cd.yml` | Deploy docs/images; points to WM train CI |

Human prompt used in CI (English):

> Please train a model on my glass classification dataset ds_glass.
> The target column is Type. Optimize for f1 and run multi-candidate
> training if available.

## Public API

```python
from hagent.world import WorldModelService

wm = WorldModelService.from_config()  # reads hagent.yaml world_model
z = wm.encode(observation)
plans = wm.plan(observation, goal)
z, z_hat, z_next, surprise = await wm.update(obs, action, next_obs)
```

Agents should only use `WorldModelService` (+ tools). Do not call encoder/predictor internals.

## Config

See `hagent.yaml` section `world_model` and `agent.planning`.
No business thresholds hard-coded in Python call sites.

## Safety

- No code-generation / `execute` tools
- Hard constraints in `agent/constraints/validator.py` before train actions
- Dynamics owned by world model, not free-form LLM plans only

## Plan executor loop + reviser

```text
coordinator → plan_executor ⇄ reviser → synthesize → END
```

| Node | Role |
|---|---|
| `plan_executor` | Run one `selected_plan` step: validate → tool → WM update → surprise |
| `reviser` | Patch/replan on validate fail, tool error, or high surprise (`max_revisions`) |

Config: `agent.planning.execute_plans`, `max_revisions`.

## Phase 5 streaming events

SSE / `stream_agent` emits:

- `route`, `phase`, `plan`, `plan_event`, `surprise`, `tool_call`, `tool_result`, `token`, `done`
- `done` includes `cost_metrics`, `plan_status`, `revision_count`

`run_agent` result also includes `execution_events`, `execution_log`, `cost_metrics`.

## Phase 6 — Multi-candidate campaigns

```text
coordinator (goal=train)
  → campaign (build N variants)
      → submit ≤ max_concurrent start_training
      → poll get_job_info
      → compare scores
      → write warm_start fact to memory
  → synthesize
```

| Piece | Path |
|---|---|
| Schema | `agent/campaign/schema.py` |
| Warm-start | `agent/campaign/warm_start.py` (WM jobs + memory) |
| Builder | `agent/campaign/builder.py` |
| Runner | `agent/campaign/runner.py` |
| Compare | `agent/campaign/compare.py` |
| Graph node | `agent/campaign/nodes.py` |

Config (`agent.campaign` in `hagent.yaml`):

- `n_job_candidates` — number of training configs
- `max_concurrent_jobs` — submit budget
- `warm_start_top_k` — past jobs / memory
- `search_algorithms`, `time_limit_options` — diversification
- `max_monitor_ticks` — graph loop safety cap

`run_agent` adds `campaign`, `campaign_status`, `evaluation` (comparison table + best job).

## Adaptive hierarchy (improved)

### Live controller

Graph node `hierarchy` (`execution/hierarchy_node.py`):

```text
coordinator → hierarchy ⇄ (analyze tools | campaign train | evaluate)
                      → synthesize
```

Config (`agent.hierarchy`):

| Key | Meaning |
|---|---|
| `live_controller` | Wire hierarchy into live graph (not metadata-only) |
| `smart_skip` | Skip leaves already satisfied by World Model |
| `abort_on_leaf_fail` | Stop hierarchy if a leaf fails |
| `templates` | Subgoal sequences per root goal_type |

### Smart-skip rules (WM-grounded)

| Leaf | Skip when |
|---|---|
| `analyze` | Dataset features (+ target) already in WM |
| `select` | `problem_type` + `metric` ready (optional past models) |
| `train` | Only if campaign already done this session |
| `evaluate` | Campaign evaluation exists, or no completed jobs |
| `monitor` | Jobs already listed in WM |

SSE events: `subgoal_start`, `subgoal_done`, `subgoal_skipped`, `hierarchy_done`.

### Hierarchy (templates)

Default train chain:

`analyze → select → train → evaluate`

Templates: `agent.hierarchy.templates` in `hagent.yaml`.

### Agent harness (recommended)

Three layers — offline modes, full multi-agent graph path, optional live API:

```bash
cd src/backend
python scripts/run_agent_harness.py --layer offline,graph --tags smoke
python scripts/run_agent_harness.py --layer graph --ids smoke_train_glass --json /tmp/h.json
python scripts/run_agent_harness.py --layer api --base-url http://localhost:5360 --token "$JWT"
```

| Layer | What it exercises |
|---|---|
| `offline` | single_shot / plan_executor / campaign / hierarchical (Phase 7) |
| `graph` | hierarchy controller + campaign + smart-skip (production path) |
| `api` | Bridge/toolkit HTTP (soft-skip if down) |

Package: `hagent/agent/harness/`. Scenarios: YAML under `harness/scenarios/`.  
Legacy: `scripts/run_eval_harness.py` → offline only (deprecated).
