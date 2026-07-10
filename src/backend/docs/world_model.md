# World Model (LeWM-inspired) — DeerFlow-AutoML Phase 4

Tham chiếu: [LeWorldModel, arXiv:2603.19312](https://arxiv.org/abs/2603.19312)

## Mapping

| LeWM | DeerFlow AutoML |
|---|---|
| Observation `o_t` (pixels) | `AutoMLObservation` (datasets/jobs/focus/phase/goal) |
| Action `a_t` | `AutoMLAction` (closed tool space) |
| Encoder `enc(o)→z` | `structured_v1` feature encoder |
| Predictor `pred(z,a)→ẑ` | `tabular_transition_v1` config deltas |
| Latent planning (CEM/MPC) | `cem_lite` planner |
| Surprise / VoE | `surprise.py` ‖ẑ − z_actual‖ |
| Offline trajectories | `TrajectoryStore` |

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
