"""
Tests cho T11 — sinh bảng LaTeX từ dữ liệu synthetic + dữ liệu HPO thật có sẵn.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent / "scripts" / "make_paper_tables.py"
_spec = importlib.util.spec_from_file_location("make_paper_tables", _SCRIPT)
mpt = importlib.util.module_from_spec(_spec)
sys.modules["make_paper_tables"] = mpt
_spec.loader.exec_module(mpt)

BENCH = Path(__file__).parent.parent / "benchmarks"


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

    def test_load_jsonl_skips_error_rows(self, tmp_path):
        p = tmp_path / "r.jsonl"
        p.write_text(
            json.dumps(_row("A", "iris", 0, 0.95))
            + "\n"
            + json.dumps(_row("A", "iris", 1, None, err="boom")),
            encoding="utf-8",
        )
        rows = mpt.load_jsonl(p)
        assert len(rows) == 1


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
            sys, "argv",
            ["make_paper_tables.py", "--out-dir", str(out)],
        )
        rc = mpt.main()
        assert rc == 0
        assert (out / "hpo_two_scales.tex").is_file()
