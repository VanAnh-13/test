"""Load scenarios from YAML packs + built-in defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from hagent.agent.harness.schema import AgentScenario, ExpectSpec


def _harness_dir() -> Path:
    return Path(__file__).resolve().parent


def load_fixture(name: str) -> Dict[str, Any]:
    """Load harness/fixtures/{name}.yaml (with or without .yaml)."""
    base = _harness_dir() / "fixtures"
    path = base / name
    if not path.suffix:
        path = base / f"{name}.yaml"
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _resolve_world_model(raw: Dict[str, Any]) -> Dict[str, Any]:
    wm = raw.get("world_model")
    if isinstance(wm, dict):
        return wm
    fixture = raw.get("world_model_fixture") or raw.get("fixture")
    if fixture:
        data = load_fixture(str(fixture))
        return data.get("world_model") or data
    return {}


def scenario_from_mapping(raw: Dict[str, Any]) -> AgentScenario:
    data = dict(raw)
    data["world_model"] = _resolve_world_model(data)
    data.pop("world_model_fixture", None)
    data.pop("fixture", None)
    return AgentScenario.from_dict(data)


def load_scenarios_from_yaml(path: Path) -> List[AgentScenario]:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    scenarios: List[AgentScenario] = []
    if isinstance(data, dict) and "scenarios" in data:
        items = data["scenarios"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "id" in data:
        items = [data]
    else:
        items = []
    for item in items:
        if isinstance(item, dict):
            scenarios.append(scenario_from_mapping(item))
    return scenarios


def load_builtin_from_eval() -> List[AgentScenario]:
    """Adapt Phase 7 EvalScenario defaults into AgentScenario."""
    from hagent.agent.eval.scenarios import default_scenarios

    out: List[AgentScenario] = []
    for s in default_scenarios():
        expect = ExpectSpec(
            goal_type=s.expect_goal_type,
            tools_called_min=s.expect_min_tools,
            has_job=s.expect_has_job if s.expect_has_job else None,
        )
        if s.expect_has_job:
            expect.tools_include = ["start_training"] if s.expect_goal_type == "train" else []
            expect.has_job = True
        out.append(
            AgentScenario(
                id=s.id,
                name=s.name,
                message=s.message,
                tags=list(s.tags),
                world_model=dict(s.world_model),
                goal=dict(s.goal),
                expect=expect,
                expect_goal_type=s.expect_goal_type,
                expect_min_tools=s.expect_min_tools,
                expect_has_job=s.expect_has_job,
                expect_metric=s.expect_metric,
            )
        )
    return out


def load_all_scenarios(
    *,
    tags: Optional[List[str]] = None,
    scenario_ids: Optional[List[str]] = None,
    packs_dir: Optional[Path] = None,
) -> List[AgentScenario]:
    scenarios = load_builtin_from_eval()
    packs = packs_dir or (_harness_dir() / "scenarios")
    if packs.exists():
        for path in sorted(packs.glob("*.yaml")):
            scenarios.extend(load_scenarios_from_yaml(path))

    # dedupe by id (YAML overrides builtin)
    by_id: Dict[str, AgentScenario] = {}
    for s in scenarios:
        by_id[s.id] = s
    scenarios = list(by_id.values())

    if scenario_ids:
        want = set(scenario_ids)
        scenarios = [s for s in scenarios if s.id in want]
    if tags:
        tagset = {t.lower() for t in tags}
        scenarios = [
            s for s in scenarios if tagset.intersection({t.lower() for t in s.tags})
        ]
    return scenarios
