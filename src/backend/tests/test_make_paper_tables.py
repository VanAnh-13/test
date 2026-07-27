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
                        "variant_sources": ["requested"],
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
                        "checkpoint_sha": "c" * 64,
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
    )


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
    def test_generates_valid_latex(self, tmp_path):
        rows = [
            _row("A", "iris", 0, 0.95),
            _row("A", "iris", 1, 0.97),
            _row("C", "iris", 0, 0.96, ext=1),
        ]
        tex = mpt.agent_matrix_table(rows)
        assert "\\begin{table}" in tex and "\\end{table}" in tex
        assert "0.9600 $\\pm$ 0.0100" in tex  # mean±std của A:iris
        assert tex.count("&") > 5
        assert "KHÔNG sửa tay" in tex

    def test_single_seed_no_pm(self):
        tex = mpt.agent_matrix_table([_row("A", "wine", 0, 0.9)])
        assert "$\\pm$" not in tex.split("wine")[1].split("\\\\")[0]

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


class TestEndToEnd:
    def test_main_writes_tables(self, tmp_path, monkeypatch):
        out = tmp_path / "tables"
        monkeypatch.setattr(
            sys,
            "argv",
            ["make_paper_tables.py", "--out-dir", str(out)],
        )
        rc = mpt.main()
        assert rc == 0
        assert (out / "hpo_two_scales.tex").is_file()
