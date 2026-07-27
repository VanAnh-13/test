"""
Tests cho T10 — runner ma trận: render yaml điều kiện, RealJobEnv job thật,
enumeration/resume. Không cần LLM.
"""

from __future__ import annotations

import asyncio
import copy
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


def matrix_config():
    path = mx.BACKEND / "benchmarks" / "agent_matrix_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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


class TestProtocolDesign:
    def test_frozen_design_is_the_main_54_cell_matrix(self):
        design = mx.build_experiment_design(matrix_config())

        assert design["protocol_version"] == "paired-meta-advice-v1"
        assert design["conditions"] == ["A", "B", "C"]
        assert design["seeds"] == [0, 1, 2]
        assert design["metric"] == "accuracy"
        assert design["search_algorithms"] == [
            "grid_search",
            "bayesian_search",
            "genetic_algorithm",
            "random_search",
            "successive_halving",
        ]
        assert (
            len(design["conditions"])
            * len(design["datasets"])
            * len(design["models"])
            * len(design["seeds"])
        ) == 54
        assert len(mx.design_sha256(matrix_config())) == 64

    def test_design_sha_changes_when_protocol_input_changes(self):
        cfg = matrix_config()
        changed = copy.deepcopy(cfg)
        changed["prompt"] += " changed"

        assert mx.design_sha256(cfg) != mx.design_sha256(changed)

    def test_non_main_condition_is_rejected(self):
        cfg = matrix_config()
        cfg["conditions"] = ["A", "B", "C_mpc"]

        with pytest.raises(mx.ProtocolError, match="conditions"):
            mx.build_experiment_design(cfg)

    def test_advice_payload_is_anonymous_and_dataset_hash_is_content_bound(self):
        from automl.search.datasets_real import load_dataset

        dataset = load_dataset("iris")
        payload = mx.anonymized_advice_payload(dataset)
        encoded = json.dumps(payload, sort_keys=True)

        assert set(payload) == {"meta_features", "metric", "search_algorithms"}
        assert set(payload["meta_features"]) == {
            "n_rows",
            "n_cols",
            "n_classes",
            "class_imbalance",
            "frac_categorical",
            "missing_frac",
            "mean_abs_skew",
        }
        assert dataset["name"] not in encoded
        assert dataset["source"] not in encoded
        assert all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in payload["meta_features"].values()
        )

        changed = dict(dataset)
        changed["X"] = dataset["X"].copy()
        changed["X"][0, 0] += 1
        assert mx.dataset_sha256(dataset) != mx.dataset_sha256(changed)


class TestPairedAdvice:
    def test_one_anonymous_invoke_returns_secret_free_provenance(self):
        from automl.search.datasets_real import load_dataset

        dataset = load_dataset("iris")
        seen = []

        def invoke(model, prompt):
            seen.append((model, prompt))
            return (
                '{"search_algorithm":"bayesian_search"}',
                {
                    "total_input_tokens": 17,
                    "total_output_tokens": 4,
                    "total_calls": 1,
                },
            )

        record = mx.request_paired_advice(
            dataset,
            model="meta-ai",
            design_sha="a" * 64,
            experiment_id="matrix-aaaaaaaaaaaaaaaa",
            invoke=invoke,
        )

        assert len(seen) == 1
        assert seen[0][0] == "meta-ai"
        assert "iris" not in seen[0][1]
        assert dataset["source"] not in seen[0][1]
        assert record["algorithm"] == "bayesian_search"
        assert record["token_usage"] == {
            "input_tokens": 17,
            "output_tokens": 4,
            "total_tokens": 21,
            "total_calls": 1,
        }
        assert len(record["prompt_sha256"]) == 64
        assert len(record["response_sha256"]) == 64
        assert "prompt" not in record
        assert "response" not in record
        assert record["cost_usd"] is None

    @pytest.mark.parametrize(
        "response",
        [
            "not-json",
            "```json\n{\"search_algorithm\":\"grid_search\"}\n```",
            "[]",
            '{"search_algorithm":"GRID_SEARCH"}',
            '{"search_algorithm":"grid_search","extra":1}',
            '{"search_algorithm":"grid_search","search_algorithm":"random_search"}',
        ],
    )
    def test_malformed_or_noncanonical_advice_fails_closed(self, response):
        from automl.search.datasets_real import load_dataset

        def invoke(_model, _prompt):
            return response, {
                "total_input_tokens": 2,
                "total_output_tokens": 1,
                "total_calls": 1,
            }

        with pytest.raises(mx.ProtocolError):
            mx.request_paired_advice(
                load_dataset("iris"),
                model="meta-ai",
                design_sha="a" * 64,
                experiment_id="matrix-aaaaaaaaaaaaaaaa",
                invoke=invoke,
            )

    @pytest.mark.parametrize(
        "usage",
        [
            {"total_input_tokens": 0, "total_output_tokens": 0, "total_calls": 1},
            {"total_input_tokens": 1, "total_output_tokens": 0, "total_calls": 0},
            {"total_input_tokens": 1, "total_output_tokens": 0, "total_calls": 2},
            {"total_input_tokens": True, "total_output_tokens": 0, "total_calls": 1},
        ],
    )
    def test_invalid_usage_fails_closed(self, usage):
        from automl.search.datasets_real import load_dataset

        with pytest.raises(mx.ProtocolError, match="usage"):
            mx.request_paired_advice(
                load_dataset("iris"),
                model="meta-ai",
                design_sha="a" * 64,
                experiment_id="matrix-aaaaaaaaaaaaaaaa",
                invoke=lambda _model, _prompt: (
                    '{"search_algorithm":"grid_search"}',
                    usage,
                ),
            )

    def test_network_error_fails_without_retry(self):
        from automl.search.datasets_real import load_dataset

        calls = 0

        def invoke(_model, _prompt):
            nonlocal calls
            calls += 1
            raise OSError("provider unavailable")

        with pytest.raises(mx.ProtocolError, match="invocation failed"):
            mx.request_paired_advice(
                load_dataset("iris"),
                model="meta-ai",
                design_sha="a" * 64,
                experiment_id="matrix-aaaaaaaaaaaaaaaa",
                invoke=invoke,
            )
        assert calls == 1


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
