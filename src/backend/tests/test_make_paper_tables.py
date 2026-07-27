"""
Tests cho T11 — sinh bảng LaTeX từ dữ liệu synthetic + dữ liệu HPO thật có sẵn.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from automl.search.datasets_real import load_dataset
from scripts import run_experiment_matrix as matrix_protocol

_SCRIPT = Path(__file__).parent.parent / "scripts" / "make_paper_tables.py"
_spec = importlib.util.spec_from_file_location("make_paper_tables", _SCRIPT)
mpt = importlib.util.module_from_spec(_spec)
sys.modules["make_paper_tables"] = mpt
_spec.loader.exec_module(mpt)

BENCH = Path(__file__).parent.parent / "benchmarks"
CONFIG = BENCH / "agent_matrix_config.yaml"
WM_MANIFEST = (
    Path(__file__).parent.parent
    / "data"
    / "world_model"
    / "outcome_ensemble_v2"
    / "manifest.json"
)
WM_CHECKPOINT_SHA = json.loads(WM_MANIFEST.read_text(encoding="utf-8"))["head"][
    "sha256"
]
FROZEN_DESIGN_SHA = "0860d3662ed8a2420aa46887cce84fdb70787c34f5b2271690976905bb4893bb"


@pytest.fixture(scope="module")
def complete_matrix_evidence():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    design = matrix_protocol.build_experiment_design(cfg)
    design_sha = matrix_protocol.design_sha256(cfg)
    assert design_sha == FROZEN_DESIGN_SHA
    experiment_id = f"matrix-{design_sha[:16]}"

    accepted = {}
    journal = []
    dataset_hashes = {}
    advice_keys = {}
    rows = []
    for dataset_index, dataset_name in enumerate(design["datasets"]):
        dataset = load_dataset(dataset_name)
        dataset_sha = matrix_protocol.dataset_sha256(dataset)
        prompt = matrix_protocol._advice_prompt(
            matrix_protocol.anonymized_advice_payload(dataset)
        )
        prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        model = design["models"][0]
        advice_key = matrix_protocol._advice_key(
            design_sha, dataset_sha, model, prompt_sha
        )
        common = {
            "record_type": "paired_meta_advice",
            "protocol_version": matrix_protocol.PROTOCOL_VERSION,
            "experiment_id": experiment_id,
            "design_sha256": design_sha,
            "advice_key": advice_key,
            "dataset_sha256": dataset_sha,
            "model": model,
            "metric": matrix_protocol.PROTOCOL_METRIC,
            "search_algorithms": list(matrix_protocol.ADVICE_ALGORITHMS),
            "prompt_sha256": prompt_sha,
        }
        pending = {**common, "status": "pending", "created_at": "2026-07-27T00:00:00Z"}
        dispatched = {
            **common,
            "status": "dispatched",
            "dispatched_at": "2026-07-27T00:00:01Z",
        }
        advice = {
            **common,
            "status": "accepted",
            "algorithm": "grid_search",
            "response_sha256": "a" * 64,
            "token_usage": {
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
                "total_calls": 1,
            },
            "cost_usd": None,
            "accepted_at": "2026-07-27T00:00:02Z",
        }
        accepted[advice_key] = advice
        journal.extend([pending, dispatched, advice])
        dataset_hashes[dataset_name] = dataset_sha
        advice_keys[(dataset_name, model)] = advice_key

        provenance_fields = (
            "experiment_id",
            "design_sha256",
            "advice_key",
            "algorithm",
            "prompt_sha256",
            "response_sha256",
            "token_usage",
        )
        for condition_index, condition in enumerate(design["conditions"]):
            for seed in design["seeds"]:
                score = (
                    0.70 + dataset_index * 0.01 + condition_index * 0.02 + seed * 0.005
                )
                job_id = f"job-{condition}-{dataset_name}-{seed}"
                rows.append(
                    {
                        "key": matrix_protocol.cell_key(
                            condition, dataset_name, model, seed
                        ),
                        "condition": condition,
                        "dataset": dataset_name,
                        "model": model,
                        "seed": seed,
                        "error": None,
                        "design_sha256": design_sha,
                        "experiment_id": experiment_id,
                        "dataset_sha256": dataset_sha,
                        "advice_provenance": {
                            field: advice[field] for field in provenance_fields
                        },
                        "requested_variant": {
                            "source": "requested",
                            "algorithm": advice["algorithm"],
                            "job_id": job_id,
                            "status": "completed",
                        },
                        "variant_sources": (
                            ["requested"]
                            if condition == "A"
                            else ["requested", "wm_planner"]
                        ),
                        "budget_score_trace": [
                            {
                                "sequence": 1,
                                "job_id": job_id,
                                "search_algorithm": advice["algorithm"],
                                "budget_seconds": 30.0,
                                "score": score,
                                "elapsed_seconds": 1.0,
                                "time_limited": False,
                            }
                        ],
                        "executed_algorithms": [advice["algorithm"]],
                        "event_types": ["campaign_done"],
                        "campaign_status": "done",
                        "best_real_score": score,
                        "n_real_jobs": 1,
                        "n_extended": 0,
                        "cost_metrics": {
                            "total_input_tokens": 12,
                            "total_output_tokens": 3,
                            "total_calls": 1,
                        },
                        "checkpoint_sha": (
                            None if condition == "A" else WM_CHECKPOINT_SHA
                        ),
                        "git_sha": "d" * 40,
                    }
                )

    return {
        "cfg": cfg,
        "rows": rows,
        "accepted": accepted,
        "journal": journal,
        "dataset_hashes": dataset_hashes,
        "advice_keys": advice_keys,
    }


def _validate(rows, evidence):
    return mpt.validate_matrix_rows(
        rows,
        cfg=evidence["cfg"],
        accepted_advices=evidence["accepted"],
        current_dataset_hashes=evidence["dataset_hashes"],
        current_advice_keys=evidence["advice_keys"],
        expected_checkpoint_sha=WM_CHECKPOINT_SHA,
    )


def _write_matrix_bundle(tmp_path, evidence, *, rows=None):
    matrix_path = tmp_path / "matrix.jsonl"
    advice_path = tmp_path / "advice.jsonl"
    matrix_rows = rows if rows is not None else evidence["rows"]
    matrix_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in matrix_rows),
        encoding="utf-8",
    )
    advice_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence["journal"]),
        encoding="utf-8",
    )
    return matrix_path, advice_path


class TestStrictMatrixGate:
    def test_accepts_exact_complete_frozen_matrix(self, complete_matrix_evidence):
        rows = _validate(
            deepcopy(complete_matrix_evidence["rows"]), complete_matrix_evidence
        )

        assert len(rows) == 54
        assert {row["key"] for row in rows} == {
            matrix_protocol.cell_key(condition, dataset, model, seed)
            for condition in complete_matrix_evidence["cfg"]["conditions"]
            for dataset in complete_matrix_evidence["cfg"]["datasets"]
            for model in complete_matrix_evidence["cfg"]["models"]
            for seed in complete_matrix_evidence["cfg"]["seeds"]
        }

    def test_rejects_one_of_fifty_four_cells(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"][:1])

        with pytest.raises(mpt.TableGateError, match="54"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_duplicate_cell_key(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[-1] = deepcopy(rows[0])

        with pytest.raises(mpt.TableGateError, match="duplicate"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_errored_cell_instead_of_filtering(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[0]["error"] = "boom"

        with pytest.raises(mpt.TableGateError, match="cell_error"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_wrong_design_sha(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[0]["design_sha256"] = "f" * 64

        with pytest.raises(mpt.TableGateError, match="design_sha_mismatch"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_missing_design_sha(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[0].pop("design_sha256")

        with pytest.raises(mpt.TableGateError, match="design_sha_mismatch"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_incomplete_seed_grid(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        row = rows[-1]
        row["seed"] = 3
        row["key"] = matrix_protocol.cell_key(
            row["condition"], row["dataset"], row["model"], row["seed"]
        )

        with pytest.raises(mpt.TableGateError, match="cell_dimension_invalid"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_score_that_disagrees_with_trace(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[0]["best_real_score"] -= 0.1

        with pytest.raises(mpt.TableGateError, match="execution trace"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_checkpoint_or_wm_source_in_condition_a(
        self, complete_matrix_evidence
    ):
        rows = deepcopy(complete_matrix_evidence["rows"])
        rows[0]["checkpoint_sha"] = WM_CHECKPOINT_SHA
        rows[0]["variant_sources"].append("wm_planner")

        with pytest.raises(mpt.TableGateError, match="condition A"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_missing_wm_evidence_in_condition_b(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        row = next(row for row in rows if row["condition"] == "B")
        row["checkpoint_sha"] = None
        row["variant_sources"] = ["requested"]

        with pytest.raises(mpt.TableGateError, match="condition B"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_checkpoint_not_bound_to_manifest(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        row = next(row for row in rows if row["condition"] == "C")
        row["checkpoint_sha"] = "e" * 64

        with pytest.raises(mpt.TableGateError, match="manifest"):
            _validate(rows, complete_matrix_evidence)

    def test_rejects_unaccepted_advice(self, complete_matrix_evidence):
        rows = deepcopy(complete_matrix_evidence["rows"])
        first_key = rows[0]["advice_provenance"]["advice_key"]
        evidence = {**complete_matrix_evidence}
        evidence["accepted"] = {
            key: value
            for key, value in complete_matrix_evidence["accepted"].items()
            if key != first_key
        }

        with pytest.raises(mpt.TableGateError, match="advice_not_accepted"):
            _validate(rows, evidence)

    @pytest.mark.parametrize(
        "content",
        [
            '{"key":"first","key":"second"}\n',
            '{"value":NaN}\n',
            "[]\n",
        ],
    )
    def test_jsonl_parser_rejects_ambiguous_or_non_object_rows(self, tmp_path, content):
        path = tmp_path / "matrix.jsonl"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(mpt.TableGateError):
            mpt.load_jsonl(path)

    def test_jsonl_parser_rejects_missing_file(self, tmp_path):
        with pytest.raises(mpt.TableGateError, match="missing"):
            mpt.load_jsonl(tmp_path / "missing.jsonl")


def _row(cond, ds, seed, score, ext=0, err=None):
    return {
        "key": f"{cond}:{ds}:m:{seed}",
        "condition": cond,
        "dataset": ds,
        "model": "m",
        "seed": seed,
        "error": err,
        "best_real_score": score,
        "n_real_jobs": 5,
        "n_extended": ext,
        "cost_metrics": {"total_input_tokens": 100, "total_output_tokens": 50},
    }


class TestAgentMatrixTable:
    def test_generates_valid_latex(self, complete_matrix_evidence):
        rows = _validate(
            deepcopy(complete_matrix_evidence["rows"]), complete_matrix_evidence
        )
        tex = mpt.agent_matrix_table(rows)
        assert "\\begin{table}" in tex and "\\end{table}" in tex
        assert "0.7050 $\\pm$ 0.0041" in tex
        assert tex.count("&") > 5
        assert "KHÔNG sửa tay" in tex
        assert FROZEN_DESIGN_SHA in tex

    def test_renderer_rejects_single_seed(self):
        with pytest.raises(mpt.TableGateError, match="54"):
            mpt.agent_matrix_table([_row("A", "wine", 0, 0.9)])

    def test_load_jsonl_preserves_error_rows_for_the_gate(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps(_row("A", "iris", 0, 0.95))
            + "\n"
            + json.dumps(_row("A", "iris", 1, None, err="boom")),
            encoding="utf-8",
        )
        rows = mpt.load_jsonl(p)
        assert len(rows) == 2
        assert rows[1]["error"] == "boom"


class TestHpoTwoScalesTable:
    def test_from_real_benchmark_files(self):
        """Dùng dữ liệu HPO THẬT đã commit trong benchmarks/."""
        small = json.loads((BENCH / "hpo_real_fair.json").read_text(encoding="utf-8"))
        large = json.loads((BENCH / "hpo_large_clean.json").read_text(encoding="utf-8"))
        tex = mpt.hpo_two_scales_table(small, large)
        assert "successive\\_halving" in tex
        assert "3.62$\\times$" in tex  # con số headline covtype phải vào bảng
        assert "\\bottomrule" in tex
        assert "single seed" in tex.lower()

    def test_validates_real_raw_grid_and_summary(self):
        small = json.loads((BENCH / "hpo_real_fair.json").read_text(encoding="utf-8"))
        large = json.loads((BENCH / "hpo_large_clean.json").read_text(encoding="utf-8"))

        mpt.validate_hpo_inputs(small, large)

    def test_rejects_summary_that_disagrees_with_raw_results(self):
        small = json.loads((BENCH / "hpo_real_fair.json").read_text(encoding="utf-8"))
        large = json.loads((BENCH / "hpo_large_clean.json").read_text(encoding="utf-8"))
        small["summary"]["grid_search"]["mean_test"] += 0.1

        with pytest.raises(mpt.TableGateError, match="summary"):
            mpt.validate_hpo_inputs(small, large)

    def test_rejects_incomplete_raw_grid(self):
        small = json.loads((BENCH / "hpo_real_fair.json").read_text(encoding="utf-8"))
        large = json.loads((BENCH / "hpo_large_clean.json").read_text(encoding="utf-8"))
        small["results"].pop()

        with pytest.raises(mpt.TableGateError, match="90"):
            mpt.validate_hpo_inputs(small, large)

    def test_rejects_missing_frozen_hpo_configuration(self):
        small = json.loads((BENCH / "hpo_real_fair.json").read_text(encoding="utf-8"))
        large = json.loads((BENCH / "hpo_large_clean.json").read_text(encoding="utf-8"))
        small.pop("budget")
        large.pop("budget")

        with pytest.raises(mpt.TableGateError, match="budget"):
            mpt.validate_hpo_inputs(small, large)

    @pytest.mark.parametrize(
        "content",
        [
            '{"summary":{"grid_search":{},"grid_search":{}}}',
            '{"summary":{"grid_search":{"mean_test":Infinity}}}',
            "[]",
        ],
    )
    def test_strict_hpo_document_parser(self, tmp_path, content):
        path = tmp_path / "hpo.json"
        path.write_text(content, encoding="utf-8")

        with pytest.raises(mpt.TableGateError):
            mpt.load_json_document(path, "HPO")


class TestEndToEnd:
    def test_missing_matrix_preserves_existing_outputs(self, tmp_path):
        out = tmp_path / "tables"
        out.mkdir()
        agent_table = out / "agent_matrix.tex"
        hpo_table = out / "hpo_two_scales.tex"
        agent_table.write_bytes(b"agent-sentinel")
        hpo_table.write_bytes(b"hpo-sentinel")

        rc = mpt.main(
            [
                "--matrix",
                str(tmp_path / "missing.jsonl"),
                "--advice",
                str(tmp_path / "missing-advice.jsonl"),
                "--config",
                str(CONFIG),
                "--hpo-small",
                str(BENCH / "hpo_real_fair.json"),
                "--hpo-large",
                str(BENCH / "hpo_large_clean.json"),
                "--out-dir",
                str(out),
            ]
        )

        assert rc == 2
        assert agent_table.read_bytes() == b"agent-sentinel"
        assert hpo_table.read_bytes() == b"hpo-sentinel"

    def test_complete_evidence_writes_both_tables_deterministically(
        self, tmp_path, complete_matrix_evidence
    ):
        matrix_path, advice_path = _write_matrix_bundle(
            tmp_path, complete_matrix_evidence
        )
        out = tmp_path / "tables"
        argv = [
            "--matrix",
            str(matrix_path),
            "--advice",
            str(advice_path),
            "--config",
            str(CONFIG),
            "--hpo-small",
            str(BENCH / "hpo_real_fair.json"),
            "--hpo-large",
            str(BENCH / "hpo_large_clean.json"),
            "--out-dir",
            str(out),
        ]

        assert mpt.main(argv) == 0
        first = {
            name: (out / name).read_bytes()
            for name in ("agent_matrix.tex", "hpo_two_scales.tex")
        }
        _write_matrix_bundle(
            tmp_path,
            complete_matrix_evidence,
            rows=list(reversed(complete_matrix_evidence["rows"])),
        )
        assert mpt.main(argv) == 0

        assert all(first.values())
        assert {
            name: (out / name).read_bytes()
            for name in ("agent_matrix.tex", "hpo_two_scales.tex")
        } == first
        assert b"wm_checkpoint_sha256=" in first["agent_matrix.tex"]
        assert b"wm_checkpoint_sha256=None" not in first["agent_matrix.tex"]

    def test_invalid_hpo_preserves_both_existing_outputs(
        self, tmp_path, complete_matrix_evidence
    ):
        matrix_path, advice_path = _write_matrix_bundle(
            tmp_path, complete_matrix_evidence
        )
        bad_small = json.loads(
            (BENCH / "hpo_real_fair.json").read_text(encoding="utf-8")
        )
        bad_small["summary"]["grid_search"]["mean_test"] += 0.1
        bad_small_path = tmp_path / "bad-small.json"
        bad_small_path.write_text(json.dumps(bad_small), encoding="utf-8")
        out = tmp_path / "tables"
        out.mkdir()
        agent_table = out / "agent_matrix.tex"
        hpo_table = out / "hpo_two_scales.tex"
        agent_table.write_bytes(b"agent-sentinel")
        hpo_table.write_bytes(b"hpo-sentinel")

        rc = mpt.main(
            [
                "--matrix",
                str(matrix_path),
                "--advice",
                str(advice_path),
                "--config",
                str(CONFIG),
                "--hpo-small",
                str(bad_small_path),
                "--hpo-large",
                str(BENCH / "hpo_large_clean.json"),
                "--out-dir",
                str(out),
            ]
        )

        assert rc == 2
        assert agent_table.read_bytes() == b"agent-sentinel"
        assert hpo_table.read_bytes() == b"hpo-sentinel"

    def test_output_failure_rolls_back_both_tables(
        self, tmp_path, complete_matrix_evidence, monkeypatch
    ):
        matrix_path, advice_path = _write_matrix_bundle(
            tmp_path, complete_matrix_evidence
        )
        out = tmp_path / "tables"
        out.mkdir()
        agent_table = out / "agent_matrix.tex"
        hpo_table = out / "hpo_two_scales.tex"
        agent_table.write_bytes(b"agent-sentinel")
        hpo_table.write_bytes(b"hpo-sentinel")
        original_write_text = Path.write_text

        def flaky_write_text(self, data, *args, **kwargs):
            if self.name == "hpo_two_scales.tex":
                raise OSError("simulated hpo write failure")
            return original_write_text(self, data, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", flaky_write_text)

        rc = mpt.main(
            [
                "--matrix",
                str(matrix_path),
                "--advice",
                str(advice_path),
                "--config",
                str(CONFIG),
                "--hpo-small",
                str(BENCH / "hpo_real_fair.json"),
                "--hpo-large",
                str(BENCH / "hpo_large_clean.json"),
                "--out-dir",
                str(out),
            ]
        )

        assert rc == 2
        assert agent_table.read_bytes() == b"agent-sentinel"
        assert hpo_table.read_bytes() == b"hpo-sentinel"
