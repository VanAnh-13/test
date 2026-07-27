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

    def test_design_sha_locks_condition_patches(self, monkeypatch):
        cfg = matrix_config()
        baseline = mx.design_sha256(cfg)
        patches = copy.deepcopy(mx.CONDITION_PATCHES)
        patches["A"]["campaign"]["wm_variant_proposal"] = True

        monkeypatch.setattr(mx, "CONDITION_PATCHES", patches)

        assert mx.design_sha256(cfg) != baseline

    def test_design_sha_locks_meta_feature_schema(self, monkeypatch):
        cfg = matrix_config()
        baseline = mx.design_sha256(cfg)

        monkeypatch.setattr(
            mx, "META_FEATURE_KEYS", mx.META_FEATURE_KEYS + ("new_feature",)
        )

        assert mx.design_sha256(cfg) != baseline

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
    def test_advice_key_binds_exact_anonymous_prompt(self):
        from automl.search.datasets_real import load_dataset

        dataset = load_dataset("iris")
        changed = dict(dataset)
        changed["meta"] = dict(dataset["meta"])
        changed["meta"]["mean_abs_skew"] += 0.25

        def invoke(_model, _prompt):
            return (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 2,
                    "total_output_tokens": 1,
                    "total_calls": 1,
                },
            )

        first = mx.request_paired_advice(
            dataset,
            model="meta-ai",
            design_sha="a" * 64,
            experiment_id="matrix-aaaaaaaaaaaaaaaa",
            invoke=invoke,
        )
        second = mx.request_paired_advice(
            changed,
            model="meta-ai",
            design_sha="a" * 64,
            experiment_id="matrix-aaaaaaaaaaaaaaaa",
            invoke=invoke,
        )

        assert first["dataset_sha256"] == second["dataset_sha256"]
        assert first["prompt_sha256"] != second["prompt_sha256"]
        assert first["advice_key"] != second["advice_key"]

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

    def test_provider_transport_disables_internal_retries(self, monkeypatch):
        from types import SimpleNamespace

        import hagent.agent.llm_config as llm_config

        seen = {}
        config = llm_config.ModelConfig(
            name="meta-ai",
            provider="openai_compatible",
            model="meta-model",
            api_key="test-key",
            base_url="http://provider.invalid/v1",
            extra={"max_retries": 9},
        )

        class FakeModel:
            def invoke(self, _prompt):
                return SimpleNamespace(
                    content='{"search_algorithm":"grid_search"}',
                    usage_metadata={"input_tokens": 2, "output_tokens": 1},
                    response_metadata={},
                )

        def build_model(provider, built_config, *_args, **_kwargs):
            seen["provider"] = provider
            seen["config"] = built_config
            return FakeModel()

        monkeypatch.setattr(llm_config, "require_model_config", lambda _name: config)
        monkeypatch.setattr(llm_config, "_build_model", build_model)

        mx._invoke_advice_model("meta-ai", "prompt")

        assert seen["provider"] == "openai_compatible"
        assert seen["config"].extra["max_retries"] == 0


class TestAdviceJournal:
    def test_one_advice_is_reused_across_conditions_and_seeds(self, tmp_path):
        from automl.search.datasets_real import load_dataset

        cells = [
            (condition, "iris", "meta-ai", seed)
            for condition in ("A", "B", "C")
            for seed in (0, 1, 2)
        ]
        sidecar = tmp_path / "advice.jsonl"
        calls = 0

        def invoke(_model, _prompt):
            nonlocal calls
            calls += 1
            return (
                '{"search_algorithm":"random_search"}',
                {
                    "total_input_tokens": 3,
                    "total_output_tokens": 2,
                    "total_calls": 1,
                },
            )

        kwargs = {
            "cells": cells,
            "datasets": {"iris": load_dataset("iris")},
            "sidecar_path": sidecar,
            "design_sha": "b" * 64,
            "experiment_id": "matrix-bbbbbbbbbbbbbbbb",
            "invoke": invoke,
        }
        first = mx.ensure_paired_advices(**kwargs)
        second = mx.ensure_paired_advices(**kwargs)

        assert calls == 1
        assert first[("iris", "meta-ai")] == second[("iris", "meta-ai")]
        records = [
            json.loads(line)
            for line in sidecar.read_text(encoding="utf-8").splitlines()
        ]
        assert [record["status"] for record in records] == [
            "pending",
            "dispatched",
            "accepted",
        ]
        assert all("prompt" not in record and "response" not in record for record in records)

    def test_pending_call_is_not_retried(self, tmp_path):
        from automl.search.datasets_real import load_dataset

        sidecar = tmp_path / "advice.jsonl"
        kwargs = {
            "cells": [("A", "iris", "meta-ai", 0)],
            "datasets": {"iris": load_dataset("iris")},
            "sidecar_path": sidecar,
            "design_sha": "c" * 64,
            "experiment_id": "matrix-cccccccccccccccc",
        }

        with pytest.raises(mx.ProtocolError, match="invocation failed"):
            mx.ensure_paired_advices(
                **kwargs,
                invoke=lambda _model, _prompt: (_ for _ in ()).throw(
                    OSError("network")
                ),
            )

        retry_calls = 0

        def retry(_model, _prompt):
            nonlocal retry_calls
            retry_calls += 1
            return (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_calls": 1,
                },
            )

        with pytest.raises(mx.ProtocolError, match="dispatched"):
            mx.ensure_paired_advices(**kwargs, invoke=retry)
        assert retry_calls == 0

    def test_pending_claim_recovers_before_dispatch_without_duplicate(
        self, tmp_path, monkeypatch
    ):
        from automl.search.datasets_real import load_dataset

        sidecar = tmp_path / "advice.jsonl"
        real_append = mx._append_jsonl

        def crash_before_dispatch(path, record):
            if record["status"] == "dispatched":
                raise OSError("crash before dispatch")
            real_append(path, record)

        monkeypatch.setattr(mx, "_append_jsonl", crash_before_dispatch)
        kwargs = {
            "cells": [("A", "iris", "meta-ai", 0)],
            "datasets": {"iris": load_dataset("iris")},
            "sidecar_path": sidecar,
            "design_sha": "c" * 64,
            "experiment_id": "matrix-cccccccccccccccc",
        }

        with pytest.raises(OSError, match="before dispatch"):
            mx.ensure_paired_advices(
                **kwargs,
                invoke=lambda _model, _prompt: pytest.fail("must not invoke"),
            )

        assert [
            json.loads(line)["status"]
            for line in sidecar.read_text(encoding="utf-8").splitlines()
        ] == ["pending"]

        monkeypatch.setattr(mx, "_append_jsonl", real_append)
        calls = 0

        def invoke(_model, _prompt):
            nonlocal calls
            calls += 1
            return (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_calls": 1,
                },
            )

        mx.ensure_paired_advices(**kwargs, invoke=invoke)

        assert calls == 1

    def test_dispatched_claim_rejects_unverified_receipt_and_retry(self, tmp_path):
        from automl.search.datasets_real import load_dataset

        dataset = load_dataset("iris")
        sidecar = tmp_path / "advice.jsonl"
        kwargs = {
            "cells": [("A", "iris", "meta-ai", 0)],
            "datasets": {"iris": dataset},
            "sidecar_path": sidecar,
            "design_sha": "c" * 64,
            "experiment_id": "matrix-cccccccccccccccc",
        }
        with pytest.raises(mx.ProtocolError, match="invocation failed"):
            mx.ensure_paired_advices(
                **kwargs,
                invoke=lambda _model, _prompt: (_ for _ in ()).throw(
                    OSError("network")
                ),
            )
        dispatched = json.loads(
            sidecar.read_text(encoding="utf-8").splitlines()[-1]
        )
        receipt = {
            key: value
            for key, value in dispatched.items()
            if key not in {"status", "dispatched_at"}
        }
        receipt.update(
            {
                "status": "accepted",
                "algorithm": "grid_search",
                "response_sha256": "f" * 64,
                "token_usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "total_calls": 1,
                },
                "cost_usd": None,
                "accepted_at": "2026-07-27T00:00:00Z",
            }
        )

        with pytest.raises(mx.ProtocolError, match="provider-authenticated"):
            mx.reconcile_dispatched_advice(sidecar, receipt)

        retry_calls = 0

        def retry(_model, _prompt):
            nonlocal retry_calls
            retry_calls += 1
            return "", {}

        with pytest.raises(mx.ProtocolError, match="dispatched"):
            mx.ensure_paired_advices(**kwargs, invoke=retry)
        assert retry_calls == 0

    def test_duplicate_accepted_record_is_corruption(self, tmp_path):
        from automl.search.datasets_real import load_dataset

        sidecar = tmp_path / "advice.jsonl"
        mx.ensure_paired_advices(
            cells=[("A", "iris", "meta-ai", 0)],
            datasets={"iris": load_dataset("iris")},
            sidecar_path=sidecar,
            design_sha="d" * 64,
            experiment_id="matrix-dddddddddddddddd",
            invoke=lambda _model, _prompt: (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 1,
                    "total_output_tokens": 1,
                    "total_calls": 1,
                },
            ),
        )
        accepted = sidecar.read_text(encoding="utf-8").splitlines()[-1]
        with sidecar.open("a", encoding="utf-8") as handle:
            handle.write(accepted + "\n")

        with pytest.raises(mx.ProtocolError, match="duplicate accepted"):
            mx.load_advice_index(sidecar)

    def test_advice_sidecar_rejects_duplicate_json_fields(self, tmp_path):
        sidecar = tmp_path / "advice.jsonl"
        sidecar.write_text(
            '{"status":"pending","status":"accepted"}\n', encoding="utf-8"
        )

        with pytest.raises(mx.ProtocolError, match="malformed"):
            mx.load_advice_index(sidecar)


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
            return info, r["job_id"]

        info, job_id = run(go())
        assert info["status"] == "completed"
        assert 0.8 < info["best_score"] <= 1.0  # iris grid thật phải > 0.8
        job = list(env.jobs.values())[0]
        assert job["best_params"]  # tham số thật từ search thật
        assert job["seconds"] > 0
        assert env.job_trace() == [
            {
                "sequence": 1,
                "job_id": job_id,
                "search_algorithm": "grid_search",
                "budget_seconds": 30.0,
                "score": job["best_score"],
                "elapsed_seconds": job["seconds"],
                "time_limited": False,
            }
        ]


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


class TestArtifactProvenance:
    def test_checkpoint_sha_is_full_sha256(self, tmp_path, monkeypatch):
        import hashlib

        import hagent.bridge.config as bridge_config

        checkpoint = tmp_path / "head.npz"
        checkpoint.write_bytes(b"checkpoint-evidence")
        monkeypatch.setattr(
            bridge_config,
            "get_world_model_config",
            lambda: {"outcome_head": {"checkpoint_path": str(checkpoint)}},
        )

        assert mx._checkpoint_sha() == hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()

    def test_git_sha_uses_full_head(self, monkeypatch):
        seen = {}

        def check_output(args, **_kwargs):
            seen["args"] = args
            return "a" * 40 + "\n"

        monkeypatch.setattr(mx.subprocess, "check_output", check_output)

        assert mx._git_sha() == "a" * 40
        assert seen["args"] == ["git", "rev-parse", "HEAD"]


class TestCellAdviceEvidence:
    def test_message_pins_advice_as_requested_constraint(self):
        from hagent.agent.planning.goal_parser import parse_goal

        message = mx.build_cell_message(
            matrix_config(),
            "iris",
            {"algorithm": "successive_halving"},
        )
        goal = parse_goal(message, known_dataset_ids=["iris"])

        assert goal["constraints"]["search_algorithm"] == "successive_halving"

    def test_reordered_requested_variant_is_bound_to_executed_job(self):
        from automl.search.datasets_real import load_dataset

        design_sha = "e" * 64
        advice = TestEnumerationAndResume.accepted_advice(design_sha)
        env = mx.RealJobEnv(
            load_dataset("iris"),
            job_cfg={
                "cv": 2,
                "time_limit": 30,
                "param_grid": {"max_depth": [3], "n_estimators": [10]},
            },
            seed=0,
        )
        started = run(
            env.invoke(
                "start_training",
                {
                    "search_algorithm": advice["algorithm"],
                    "dataset_id": "iris",
                    "time_limit": 30,
                },
            )
        )
        job_id = started["job_id"]
        result = {
            "campaign": {
                "variants": [
                    {
                        "source": "diversified",
                        "params": {"search_algorithm": "random_search"},
                        "job_id": "other",
                        "status": "completed",
                    },
                    {
                        "source": "requested",
                        "params": {"search_algorithm": advice["algorithm"]},
                        "job_id": job_id,
                        "status": "completed",
                    },
                ]
            },
            "execution_events": [{"type": "campaign_done"}],
        }

        evidence = mx.build_cell_evidence(result, env, advice)

        assert evidence["requested_variant"]["job_id"] == job_id
        assert evidence["executed_algorithms"] == [advice["algorithm"]]
        assert evidence["event_types"] == ["campaign_done"]

    def test_unexecuted_advice_fails_closed(self):
        from automl.search.datasets_real import load_dataset

        advice = TestEnumerationAndResume.accepted_advice("e" * 64)
        env = mx.RealJobEnv(load_dataset("iris"), job_cfg={}, seed=0)
        result = {
            "campaign": {
                "variants": [
                    {
                        "source": "requested",
                        "params": {"search_algorithm": advice["algorithm"]},
                        "job_id": "missing",
                        "status": "completed",
                    }
                ]
            }
        }

        with pytest.raises(mx.ProtocolError, match="executed"):
            mx.build_cell_evidence(result, env, advice)

    def test_run_cell_clears_invoker_when_message_rendering_fails(self, tmp_path):
        import hagent.agent.execution.tool_runner as tool_runner

        design_sha = "e" * 64
        advice = TestEnumerationAndResume.accepted_advice(design_sha)
        cfg = matrix_config()
        cfg["prompt"] = "Train {dataset} target {target} with {missing}"

        try:
            with pytest.raises(KeyError, match="missing"):
                mx.run_cell(
                    "A",
                    "iris",
                    "meta-ai",
                    0,
                    cfg=cfg,
                    scratch_dir=tmp_path,
                    advice=advice,
                    design_sha=design_sha,
                    experiment_id=advice["experiment_id"],
                    agent_runner=lambda *_args, **_kwargs: pytest.fail(
                        "agent must not run"
                    ),
                )

            assert tool_runner._tool_invoker is None
        finally:
            tool_runner.set_tool_invoker(None)

    def test_run_cell_emits_complete_protocol_evidence(self, tmp_path):
        from hagent.agent.execution.tool_runner import invoke_tool

        design_sha = "e" * 64
        advice = TestEnumerationAndResume.accepted_advice(design_sha)
        cfg = matrix_config()
        cfg["job"] = {
            "cv": 2,
            "time_limit": 30,
            "param_grid": {"max_depth": [3], "n_estimators": [10]},
        }

        async def agent_runner(message, **_kwargs):
            assert advice["algorithm"] in message
            started = await invoke_tool(
                "start_training",
                {
                    "search_algorithm": advice["algorithm"],
                    "dataset_id": "iris",
                    "time_limit": 30,
                },
            )
            return {
                "response": "done",
                "campaign_status": "done",
                "campaign": {
                    "status": "done",
                    "variants": [
                        {
                            "source": "requested",
                            "params": {"search_algorithm": advice["algorithm"]},
                            "job_id": started["job_id"],
                            "status": "completed",
                        }
                    ],
                    "extension_rounds": 0,
                },
                "execution_events": [{"type": "campaign_done"}],
                "cost_metrics": {
                    "total_input_tokens": 8,
                    "total_output_tokens": 2,
                    "total_calls": 1,
                },
            }

        row = mx.run_cell(
            "A",
            "iris",
            "meta-ai",
            0,
            cfg=cfg,
            scratch_dir=tmp_path,
            advice=advice,
            design_sha=design_sha,
            experiment_id=advice["experiment_id"],
            agent_runner=agent_runner,
        )

        assert row["error"] is None
        assert mx.validate_result_evidence(
            row, design_sha, {advice["advice_key"]: advice}
        ) == []


class TestEnumerationAndResume:

    def test_cell_key_format(self):
        assert mx.cell_key("A", "iris", "m", 0) == "A:iris:m:0"

    def test_legacy_rows_are_not_resumed(self, tmp_path):
        out = tmp_path / "results.jsonl"
        rows = [
            {"key": "A:iris:m:0", "error": None},
            {"key": "A:wine:m:0", "error": "boom"},  # lỗi → phải chạy lại
        ]
        out.write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )
        done = mx.migrate_rejected_rows(
            out, tmp_path / "rejected.jsonl",
            design_sha="e" * 64, accepted_advices={}
        )
        assert done == set()
        assert not out.exists()
        assert len((tmp_path / "rejected.jsonl").read_text().splitlines()) == 2

    @staticmethod
    def accepted_advice(design_sha):
        from automl.search.datasets_real import load_dataset

        return mx.request_paired_advice(
            load_dataset("iris"),
            model="meta-ai",
            design_sha=design_sha,
            experiment_id=f"matrix-{design_sha[:16]}",
            invoke=lambda _model, _prompt: (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 5,
                    "total_output_tokens": 2,
                    "total_calls": 1,
                },
            ),
        )

    @staticmethod
    def complete_row(advice):
        provenance_keys = {
            "experiment_id",
            "design_sha256",
            "advice_key",
            "algorithm",
            "prompt_sha256",
            "response_sha256",
            "token_usage",
        }
        return {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
            "error": None,
            "design_sha256": advice["design_sha256"],
            "experiment_id": advice["experiment_id"],
            "dataset_sha256": advice["dataset_sha256"],
            "advice_provenance": {
                key: advice[key] for key in provenance_keys
            },
            "campaign_status": "done",
            "variant_sources": ["requested"],
            "requested_variant": {
                "source": "requested",
                "algorithm": advice["algorithm"],
                "job_id": "real_1",
                "status": "completed",
            },
            "cost_metrics": {
                "total_input_tokens": 11,
                "total_output_tokens": 3,
                "total_calls": 1,
            },
            "n_real_jobs": 1,
            "budget_score_trace": [
                {
                    "sequence": 1,
                    "job_id": "real_1",
                    "search_algorithm": advice["algorithm"],
                    "budget_seconds": 60.0,
                    "score": 0.95,
                    "elapsed_seconds": 1.5,
                    "time_limited": False,
                }
            ],
            "executed_algorithms": [advice["algorithm"]],
            "event_types": [],
        }

    def test_resume_accepts_only_complete_matching_evidence(self):
        design_sha = "e" * 64
        advice = self.accepted_advice(design_sha)
        row = self.complete_row(advice)

        partition = mx.partition_resume_rows(
            [row], design_sha, {advice["advice_key"]: advice}
        )

        assert partition["done"] == {"A:iris:meta-ai:0"}
        assert partition["rejected"] == []

    def test_changed_current_dataset_hash_is_rejected(self, tmp_path):
        design_sha = "e" * 64
        advice = self.accepted_advice(design_sha)
        row = self.complete_row(advice)
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        results.write_text(json.dumps(row) + "\n", encoding="utf-8")
        changed_hash = "f" * 64
        assert changed_hash != row["dataset_sha256"]

        done = mx.migrate_rejected_rows(
            results,
            rejected,
            design_sha=design_sha,
            accepted_advices={advice["advice_key"]: advice},
            current_dataset_hashes={"iris": changed_hash},
        )

        assert done == set()
        assert not results.exists()
        record = json.loads(rejected.read_text(encoding="utf-8"))
        assert "dataset_content_mismatch" in record["reason_codes"]

    def test_changed_current_advice_prompt_is_not_resumed(self):
        import hashlib

        from automl.search.datasets_real import load_dataset

        design_sha = "e" * 64
        dataset = load_dataset("iris")
        advice = self.accepted_advice(design_sha)
        row = self.complete_row(advice)
        changed = dict(dataset)
        changed["meta"] = dict(dataset["meta"])
        changed["meta"]["mean_abs_skew"] += 0.25
        assert mx.dataset_sha256(changed) == row["dataset_sha256"]
        prompt_sha = hashlib.sha256(
            mx._advice_prompt(
                mx.anonymized_advice_payload(changed)
            ).encode("utf-8")
        ).hexdigest()
        current_key = mx._advice_key(
            design_sha,
            row["dataset_sha256"],
            "meta-ai",
            prompt_sha,
        )
        assert current_key != advice["advice_key"]

        partition = mx.partition_resume_rows(
            [row],
            design_sha,
            {advice["advice_key"]: advice},
            current_dataset_hashes={"iris": row["dataset_sha256"]},
            current_advice_keys={("iris", "meta-ai"): current_key},
        )

        assert partition["done"] == set()
        assert "advice_prompt_mismatch" in partition["rejected"][0]["reason_codes"]

    @pytest.mark.parametrize(
        ("case", "reason"),
        [
            ("wrong_design", "design_sha_mismatch"),
            ("error", "cell_error"),
            ("zero_usage", "cell_usage_invalid"),
            ("missing_trace", "budget_score_trace_invalid"),
            ("algorithm_conflict", "advised_algorithm_not_executed"),
        ],
    )
    def test_incomplete_or_conflicting_evidence_is_not_resumed(self, case, reason):
        design_sha = "e" * 64
        advice = self.accepted_advice(design_sha)
        row = self.complete_row(advice)
        if case == "wrong_design":
            row["design_sha256"] = "f" * 64
        elif case == "error":
            row["error"] = "failed"
        elif case == "zero_usage":
            row["cost_metrics"]["total_input_tokens"] = 0
            row["cost_metrics"]["total_output_tokens"] = 0
            row["cost_metrics"]["total_calls"] = 0
        elif case == "missing_trace":
            row.pop("budget_score_trace")
        else:
            row["executed_algorithms"] = ["random_search"]

        partition = mx.partition_resume_rows(
            [row], design_sha, {advice["advice_key"]: advice}
        )

        assert partition["done"] == set()
        assert reason in partition["rejected"][0]["reason_codes"]

    def test_legacy_zero_call_row_is_migrated_idempotently(self, tmp_path):
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        legacy = {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
            "error": None,
            "cost_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            },
        }
        results.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        first = mx.migrate_rejected_rows(
            results, rejected, design_sha="e" * 64, accepted_advices={}
        )
        before = rejected.read_text(encoding="utf-8")
        second = mx.migrate_rejected_rows(
            results, rejected, design_sha="e" * 64, accepted_advices={}
        )

        assert first == second == set()
        assert not results.exists()
        assert not results.with_name("results.jsonl.tmp").exists()
        assert rejected.read_text(encoding="utf-8") == before
        rejection = json.loads(before)
        assert rejection["key"] == legacy["key"]
        assert "cell_usage_invalid" in rejection["reason_codes"]

    def test_forged_rejection_record_aborts_without_deleting_source(self, tmp_path):
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        legacy = {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
            "error": None,
            "cost_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            },
        }
        results.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        reasons = mx.validate_result_evidence(legacy, "e" * 64, {})
        forged = mx._rejection_record(legacy, reasons)
        forged["row_sha256"] = "0" * 64
        rejected.write_text(json.dumps(forged) + "\n", encoding="utf-8")
        before = results.read_bytes()

        with pytest.raises(mx.ProtocolError, match="rejection line"):
            mx.migrate_rejected_rows(
                results,
                rejected,
                design_sha="e" * 64,
                accepted_advices={},
                current_dataset_hashes={"iris": "f" * 64},
            )

        assert results.read_bytes() == before
        assert not results.with_name("results.jsonl.tmp").exists()

    def test_rejection_projection_drops_unapproved_nested_text(self):
        marker = "do-not-persist-secret"
        row = {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
            "error": marker,
            "prompt": marker,
            "advice_provenance": {
                "advice_key": "a" * 64,
                "algorithm": "grid_search",
                "prompt": marker,
                "response": marker,
            },
        }

        record = mx._rejection_record(row, ["cell_error"])

        assert marker not in json.dumps(record, sort_keys=True)
        mx._validate_rejection_record(record, line_number=1)

    def test_rejection_id_binds_full_original_row(self):
        base = {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
        }
        first_row = {
            **base,
            "error": "ProtocolError: first internal failure",
            "budget_score_trace": [{"private_detail": "first"}],
        }
        second_row = {
            **base,
            "error": "ProtocolError: second internal failure",
            "budget_score_trace": [{"private_detail": "second"}],
        }

        first = mx._rejection_record(first_row, ["cell_error"])
        second = mx._rejection_record(second_row, ["cell_error"])

        assert first["row"] == second["row"]
        assert first["row_sha256"] != second["row_sha256"]
        assert first["rejection_id"] != second["rejection_id"]

    @pytest.mark.parametrize(
        "line",
        [
            '{"rejection_id":"a","rejection_id":"b"}',
            '{"value":NaN}',
        ],
    )
    def test_rejection_sidecar_rejects_non_strict_json_without_new_rows(
        self, tmp_path, line
    ):
        rejected = tmp_path / "rejected.jsonl"
        rejected.write_text(line + "\n", encoding="utf-8")

        with pytest.raises(mx.ProtocolError, match="malformed"):
            mx.migrate_rejected_rows(
                tmp_path / "results.jsonl",
                rejected,
                design_sha="e" * 64,
                accepted_advices={},
            )

    def test_migration_recovers_zero_byte_temp_prefix(self, tmp_path):
        design_sha = "e" * 64
        advice = self.accepted_advice(design_sha)
        kept = self.complete_row(advice)
        legacy = {
            "key": "A:wine:meta-ai:0",
            "condition": "A",
            "dataset": "wine",
            "model": "meta-ai",
            "seed": 0,
            "error": None,
            "cost_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            },
        }
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        temp = results.with_name("results.jsonl.tmp")
        results.write_text(
            json.dumps(kept) + "\n" + json.dumps(legacy) + "\n",
            encoding="utf-8",
        )
        temp.write_bytes(b"")

        done = mx.migrate_rejected_rows(
            results,
            rejected,
            design_sha=design_sha,
            accepted_advices={advice["advice_key"]: advice},
        )

        assert done == {kept["key"]}
        assert not temp.exists()
        remaining = [
            json.loads(line)
            for line in results.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["key"] for row in remaining] == [kept["key"]]
        rejection = json.loads(rejected.read_text(encoding="utf-8"))
        assert rejection["key"] == legacy["key"]

    def test_migration_recovers_complete_temp_after_replace_failure(
        self, tmp_path, monkeypatch
    ):
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        legacy = {
            "key": "A:iris:meta-ai:0",
            "condition": "A",
            "dataset": "iris",
            "model": "meta-ai",
            "seed": 0,
            "error": None,
            "cost_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            },
        }
        results.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        real_replace = mx.os.replace

        def fail_replace(_source, _target):
            raise OSError("replace interrupted")

        monkeypatch.setattr(mx.os, "replace", fail_replace)
        with pytest.raises(OSError, match="interrupted"):
            mx.migrate_rejected_rows(
                results,
                rejected,
                design_sha="e" * 64,
                accepted_advices={},
                current_dataset_hashes={"iris": "f" * 64},
            )

        assert results.exists()
        assert results.with_name("results.jsonl.tmp").exists()
        monkeypatch.setattr(mx.os, "replace", real_replace)

        done = mx.migrate_rejected_rows(
            results,
            rejected,
            design_sha="e" * 64,
            accepted_advices={},
            current_dataset_hashes={"iris": "f" * 64},
        )

        assert done == set()
        assert not results.exists()
        assert not results.with_name("results.jsonl.tmp").exists()

    def test_duplicate_result_key_aborts_before_writes(self, tmp_path):
        design_sha = "e" * 64
        advice = self.accepted_advice(design_sha)
        row = self.complete_row(advice)
        results = tmp_path / "results.jsonl"
        rejected = tmp_path / "rejected.jsonl"
        results.write_text(
            "\n".join(json.dumps(row) for _ in range(2)) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(mx.ProtocolError, match="duplicate result key"):
            mx.migrate_rejected_rows(
                results,
                rejected,
                design_sha=design_sha,
                accepted_advices={advice["advice_key"]: advice},
            )

        assert not rejected.exists()
        assert not results.with_name("results.jsonl.tmp").exists()


class TestMatrixCli:
    @staticmethod
    def write_config(tmp_path, *, unknown_dataset=False):
        cfg = matrix_config()
        if unknown_dataset:
            cfg["datasets"][0] = "not_loaded_in_dry_run"
        cfg["output"] = str(tmp_path / "results.jsonl")
        cfg["advice_output"] = str(tmp_path / "advice.jsonl")
        cfg["rejected_output"] = str(tmp_path / "rejected.jsonl")
        path = tmp_path / "matrix.yaml"
        path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return path, cfg

    def test_dry_run_is_call_free_and_does_not_migrate(self, tmp_path):
        config_path, cfg = self.write_config(tmp_path, unknown_dataset=True)
        results = Path(cfg["output"])
        legacy = {
            "key": "A:iris:meta-ai:0",
            "error": None,
            "cost_metrics": {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_calls": 0,
            },
        }
        results.write_text(json.dumps(legacy) + "\n", encoding="utf-8")
        before = results.read_bytes()

        exit_code = mx.main(["--config", str(config_path), "--dry-run"])

        assert exit_code == 0
        assert results.read_bytes() == before
        assert not Path(cfg["advice_output"]).exists()
        assert not Path(cfg["rejected_output"]).exists()

    def test_live_preflight_hashes_all_datasets_before_resume(
        self, tmp_path, monkeypatch
    ):
        import numpy as np

        import automl.search.datasets_real as datasets_real
        import hagent.agent.llm_config as llm_config

        config_path, cfg = self.write_config(tmp_path)
        loaded = []
        captured = {}

        def load_dataset(name):
            loaded.append(name)
            return {
                "X": np.asarray([[1.0], [2.0]], dtype=float),
                "y": np.asarray([0.0, 1.0], dtype=float),
                "meta": {
                    "n_rows": 2,
                    "n_cols": 1,
                    "n_classes": 2,
                    "class_imbalance": 0.0,
                    "frac_categorical": 0.0,
                    "missing_frac": 0.0,
                    "mean_abs_skew": 0.0,
                },
            }

        def migrate(*_args, current_dataset_hashes=None, **_kwargs):
            captured.update(current_dataset_hashes or {})
            return {"A:iris:meta-ai:0"}

        monkeypatch.setattr(datasets_real, "load_dataset", load_dataset)
        monkeypatch.setattr(llm_config, "require_model_config", lambda _name: object())
        monkeypatch.setattr(mx, "migrate_rejected_rows", migrate)

        exit_code = mx.main(
            [
                "--config",
                str(config_path),
                "--only",
                "A:iris:meta-ai:0",
            ],
            advice_invoke=lambda *_args: pytest.fail("advice must not run"),
            agent_runner=lambda *_args, **_kwargs: pytest.fail("agent must not run"),
        )

        assert exit_code == 0
        assert set(loaded) == set(cfg["datasets"])
        assert set(captured) == set(cfg["datasets"])
        assert all(len(value) == 64 for value in captured.values())

    @pytest.mark.parametrize(
        "only",
        [
            "C_mpc:iris:meta-ai:0",
            "A:iris:meta-ai:99",
            "A:iris:typo-model:0",
        ],
    )
    def test_only_must_be_a_member_of_frozen_design(self, tmp_path, only):
        config_path, cfg = self.write_config(tmp_path)

        exit_code = mx.main(
            ["--config", str(config_path), "--dry-run", "--only", only]
        )

        assert exit_code == 2
        assert not Path(cfg["output"]).exists()
        assert not Path(cfg["advice_output"]).exists()
        assert not Path(cfg["rejected_output"]).exists()

    def test_live_only_uses_one_advice_and_resumes_without_new_calls(self, tmp_path):
        from hagent.agent.execution.tool_runner import invoke_tool

        config_path, cfg = self.write_config(tmp_path)
        cfg["job"] = {
            "cv": 2,
            "time_limit": 30,
            "param_grid": {"max_depth": [3], "n_estimators": [10]},
        }
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        advice_calls = 0
        agent_calls = 0

        def advice_invoke(_model, _prompt):
            nonlocal advice_calls
            advice_calls += 1
            return (
                '{"search_algorithm":"grid_search"}',
                {
                    "total_input_tokens": 4,
                    "total_output_tokens": 1,
                    "total_calls": 1,
                },
            )

        async def agent_runner(_message, **_kwargs):
            nonlocal agent_calls
            agent_calls += 1
            started = await invoke_tool(
                "start_training",
                {
                    "search_algorithm": "grid_search",
                    "dataset_id": "iris",
                    "time_limit": 30,
                },
            )
            return {
                "response": "done",
                "campaign_status": "done",
                "campaign": {
                    "status": "done",
                    "variants": [
                        {
                            "source": "requested",
                            "params": {"search_algorithm": "grid_search"},
                            "job_id": started["job_id"],
                            "status": "completed",
                        }
                    ],
                    "extension_rounds": 0,
                },
                "execution_events": [{"type": "campaign_done"}],
                "cost_metrics": {
                    "total_input_tokens": 6,
                    "total_output_tokens": 2,
                    "total_calls": 1,
                },
            }

        argv = [
            "--config",
            str(config_path),
            "--only",
            "A:iris:meta-ai:0",
        ]
        first = mx.main(
            argv, advice_invoke=advice_invoke, agent_runner=agent_runner
        )
        second = mx.main(
            argv, advice_invoke=advice_invoke, agent_runner=agent_runner
        )

        assert first == second == 0
        assert advice_calls == agent_calls == 1
        row = json.loads(Path(cfg["output"]).read_text(encoding="utf-8"))
        advice_state = mx.load_advice_index(Path(cfg["advice_output"]))
        assert mx.validate_result_evidence(
            row, mx.design_sha256(cfg), advice_state["accepted"]
        ) == []
        assert not Path(cfg["rejected_output"]).exists()
