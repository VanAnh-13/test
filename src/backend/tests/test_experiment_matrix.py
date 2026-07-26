"""
Tests cho T10 — runner ma trận: render yaml điều kiện, RealJobEnv job thật,
enumeration/resume. Không cần LLM.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

_SCRIPT = Path(__file__).parent.parent / "scripts" / "run_experiment_matrix.py"
_spec = importlib.util.spec_from_file_location("run_experiment_matrix", _SCRIPT)
mx = importlib.util.module_from_spec(_spec)
sys.modules["run_experiment_matrix"] = mx
_spec.loader.exec_module(mx)


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestConditionYaml:
    def test_condition_a_disables_everything(self, tmp_path):
        p = mx.render_condition_yaml("A", tmp_path)
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
        camp = cfg["agent"]["campaign"]
        assert camp["wm_variant_proposal"] is False
        assert camp["wm_rank_variants"] is False
        assert camp["surprise_extension"]["enabled"] is False
        assert "__disabled__" in cfg["world_model"]["outcome_head"]["checkpoint_path"]

    def test_condition_c_enables_extension(self, tmp_path):
        cfg = yaml.safe_load(
            mx.render_condition_yaml("C", tmp_path).read_text(encoding="utf-8")
        )
        camp = cfg["agent"]["campaign"]
        assert camp["wm_variant_proposal"] is True
        assert camp["surprise_extension"]["enabled"] is True
        # checkpoint giữ nguyên đường v2 (không bị patch A dây sang)
        assert "v2" in cfg["world_model"]["outcome_head"]["checkpoint_path"]

    def test_condition_cmpc_switches_planner(self, tmp_path):
        cfg = yaml.safe_load(
            mx.render_condition_yaml("C_mpc", tmp_path).read_text(encoding="utf-8")
        )
        assert cfg["world_model"]["campaign_planner"]["backend"] == "cem_mpc_v1"

    def test_unknown_condition_raises(self, tmp_path):
        with pytest.raises(ValueError):
            mx.render_condition_yaml("Z", tmp_path)

    def test_base_yaml_untouched_keys_survive(self, tmp_path):
        cfg = yaml.safe_load(
            mx.render_condition_yaml("B", tmp_path).read_text(encoding="utf-8")
        )
        # các phần không patch vẫn nguyên (llm models, encoder...)
        assert cfg["llm"]["models"]
        assert cfg["world_model"]["encoder"]["dim"] == 64


class TestRealJobEnv:
    def test_real_training_job_on_iris(self):
        from automl.search.datasets_real import load_dataset

        env = mx.RealJobEnv(
            load_dataset("iris"),
            job_cfg={"cv": 3, "time_limit": 30,
                     "param_grid": {"max_depth": [3, 5], "n_estimators": [10]}},
            seed=0,
        )

        async def go():
            r = await env.invoke(
                "start_training",
                {"search_algorithm": "grid_search", "dataset_id": "iris"},
            )
            info = await env.invoke("get_job_info", {"job_id": r["job_id"]})
            return info

        info = run(go())
        assert info["status"] == "completed"
        assert 0.8 < info["best_score"] <= 1.0  # iris grid thật phải > 0.8
        job = list(env.jobs.values())[0]
        assert job["best_params"]  # tham số thật từ search thật
        assert job["seconds"] > 0

    def test_dataset_info_from_registry(self):
        from automl.search.datasets_real import load_dataset

        env = mx.RealJobEnv(load_dataset("wine"), job_cfg={}, seed=0)
        info = run(env.invoke("get_dataset_info", {"dataset_id": "wine"}))
        assert info["n_rows"] == 178
        assert info["target"] == "target"

    def test_unknown_job_error(self):
        from automl.search.datasets_real import load_dataset

        env = mx.RealJobEnv(load_dataset("iris"), job_cfg={}, seed=0)
        info = run(env.invoke("get_job_info", {"job_id": "nope"}))
        assert info.get("error")


class TestEnumerationAndResume:
    def test_cell_key_format(self):
        assert mx.cell_key("A", "iris", "m", 0) == "A:iris:m:0"

    def test_resume_skips_only_successful_rows(self, tmp_path):
        out = tmp_path / "results.jsonl"
        rows = [
            {"key": "A:iris:m:0", "error": None},
            {"key": "A:wine:m:0", "error": "boom"},  # lỗi → phải chạy lại
        ]
        out.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        done = set()
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("error"):
                done.add(row["key"])
        assert done == {"A:iris:m:0"}
