#!/usr/bin/env python3
"""
Student Performance — realistic CI/CD end-to-end runner.

Layers (default: harness + mock-api):
  1. harness   — offline + graph agent harness (tags: student)
  2. mock-api  — Mock HAutoML HTTP flow: list → info → train RF/XGB/SVR → job info
  3. agent     — optional live DeerFlow agent (needs Ollama / real LLM)

Usage (from src/backend):
  python scripts/run_student_performance_e2e.py
  python scripts/run_student_performance_e2e.py --layers harness,mock-api
  python scripts/run_student_performance_e2e.py --layers agent --user-id ci_user
  python scripts/run_student_performance_e2e.py --json results/student_e2e.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# Canonical Student Performance expectations (aligned with mock_hautoml_server)
STUDENT_DATASET_ID = "ds_student_001"
STUDENT_TARGET = "G3"
STUDENT_N_ROWS = 395
STUDENT_N_COLS = 33
STUDENT_MODELS = [
    "RandomForestRegressor",
    "XGBRegressor",
    "SVR",
]
# XGB has lowest RMSE in the mock training table
EXPECTED_BEST_MODEL = "XGBRegressor"
EXPECTED_BEST_RMSE_MAX = 1.70
EXPECTED_MODEL_RMSE = {
    "RandomForestRegressor": 1.82,
    "XGBRegressor": 1.65,
    "SVR": 2.15,
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayerReport:
    layer: str
    ok: bool
    checks: List[CheckResult] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "ok": self.ok,
            "elapsed_seconds": self.elapsed_seconds,
            "checks": [c.to_dict() for c in self.checks],
            "extra": self.extra,
        }


def _http_json(
    method: str,
    url: str,
    body: Optional[dict] = None,
    timeout: float = 30.0,
) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _assert(checks: List[CheckResult], name: str, ok: bool, detail: str = "", **data):
    checks.append(CheckResult(name=name, ok=bool(ok), detail=detail, data=dict(data)))
    status = "✓" if ok else "✗"
    print(f"  {status} {name}" + (f" — {detail}" if detail else ""))


# ── Layer: harness ───────────────────────────────────────


async def run_harness_layer() -> LayerReport:
    from hagent.agent.harness import report_markdown, run_harness_suite

    print("\n═══ Layer: harness (offline + graph, tags=student) ═══")
    t0 = time.time()
    report = await run_harness_suite(
        layers=["offline", "graph"],
        offline_modes=["single_shot", "plan_executor", "campaign", "hierarchical"],
        tags=["student"],
    )
    elapsed = time.time() - t0
    checks: List[CheckResult] = []

    n = int(report.get("n") or 0)
    n_failed = int(report.get("n_failed") or 0)
    _assert(checks, "harness_runs", n > 0, f"n={n}")
    _assert(checks, "harness_all_pass", n_failed == 0, f"failed={n_failed}/{n}")

    # Must cover train multi-model scenario
    ids = {r.get("scenario_id") for r in report.get("results") or []}
    _assert(
        checks,
        "has_student_train_multi",
        "student_train_multi" in ids,
        f"ids={sorted(ids)}",
    )

    train_rows = [
        r
        for r in report.get("results") or []
        if r.get("scenario_id") == "student_train_multi"
    ]
    train_ok = all(r.get("success") for r in train_rows) if train_rows else False
    _assert(
        checks,
        "student_train_multi_ok",
        train_ok,
        f"rows={len(train_rows)}",
    )

    ok = all(c.ok for c in checks)
    print(report_markdown(report))
    return LayerReport(
        layer="harness",
        ok=ok,
        checks=checks,
        elapsed_seconds=round(elapsed, 3),
        extra={"n": n, "n_failed": n_failed, "report": report},
    )


# ── Layer: mock-api ──────────────────────────────────────


def run_mock_api_layer(
    *,
    base_url: str,
    models: Sequence[str],
    start_server: bool,
    port: int,
) -> LayerReport:
    print(f"\n═══ Layer: mock-api ({base_url}) ═══")
    checks: List[CheckResult] = []
    t0 = time.time()
    proc: Optional[subprocess.Popen] = None
    extra: Dict[str, Any] = {}

    try:
        if start_server:
            script = BACKEND / "scripts" / "mock_hautoml_server.py"
            proc = subprocess.Popen(
                [sys.executable, str(script), "--port", str(port)],
                cwd=str(BACKEND),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            # Wait for health
            ready = False
            for _ in range(40):
                try:
                    _http_json("GET", f"{base_url}/home", timeout=2)
                    ready = True
                    break
                except Exception:
                    time.sleep(0.25)
            _assert(checks, "mock_server_ready", ready, f"port={port}")
            if not ready:
                return LayerReport(
                    layer="mock-api",
                    ok=False,
                    checks=checks,
                    elapsed_seconds=round(time.time() - t0, 3),
                )

        # 1) Health
        try:
            health = _http_json("GET", f"{base_url}/home")
            _assert(
                checks,
                "health",
                health.get("status") == "ok",
                str(health)[:120],
            )
        except Exception as exc:
            _assert(checks, "health", False, str(exc))
            return LayerReport(
                layer="mock-api",
                ok=False,
                checks=checks,
                elapsed_seconds=round(time.time() - t0, 3),
            )

        # 2) List datasets
        listed = _http_json(
            "POST",
            f"{base_url}/get-list-data-by-userid?id=student_ci",
        )
        datasets = listed.get("datasets") if isinstance(listed, dict) else listed
        if not isinstance(datasets, list):
            datasets = []
        has_student = any(
            (d.get("id") or d.get("_id")) == STUDENT_DATASET_ID for d in datasets
        )
        _assert(
            checks,
            "list_has_student",
            has_student,
            f"count={len(datasets)}",
            datasets=datasets,
        )

        # 3) Dataset info
        info = _http_json("GET", f"{base_url}/get-data-info?id={STUDENT_DATASET_ID}")
        _assert(
            checks,
            "info_target_G3",
            info.get("target") == STUDENT_TARGET,
            f"target={info.get('target')}",
        )
        _assert(
            checks,
            "info_shape",
            info.get("n_rows") == STUDENT_N_ROWS and info.get("n_cols") == STUDENT_N_COLS,
            f"shape={info.get('n_rows')}x{info.get('n_cols')}",
        )
        feats = info.get("features") or []
        _assert(
            checks,
            "info_features",
            isinstance(feats, list) and "G1" in feats and "G2" in feats,
            f"n_features={len(feats) if isinstance(feats, list) else 0}",
        )

        # 4) Available regression models
        avail = _http_json("GET", f"{base_url}/api/v1/available-models/regression")
        model_names = []
        for m in avail.get("models") or []:
            if isinstance(m, dict):
                model_names.append(m.get("name"))
            else:
                model_names.append(str(m))
        for m in models:
            _assert(
                checks,
                f"available_model_{m}",
                m in model_names,
                f"available={model_names}",
            )

        # 5) Train multi-model
        train_body = {
            "models": list(models),
            "target_column": STUDENT_TARGET,
            "problem_type": "regression",
            "metric": "rmse",
        }
        train_url = (
            f"{base_url}/train-from-requestbody-json/"
            f"?userId=student_ci&id_data={STUDENT_DATASET_ID}"
        )
        job = _http_json("POST", train_url, body=train_body)
        job_id = job.get("job_id") or job.get("id")
        _assert(checks, "train_returns_job_id", bool(job_id), f"job={job_id}")
        _assert(
            checks,
            "train_status_completed",
            job.get("status") in ("completed", "done", "success", "starting"),
            f"status={job.get('status')}",
        )

        best_model = job.get("best_model")
        best_score = job.get("best_score")
        _assert(
            checks,
            "best_model_xgb",
            best_model == EXPECTED_BEST_MODEL,
            f"best_model={best_model}",
        )
        try:
            score_f = float(best_score)
        except (TypeError, ValueError):
            score_f = 999.0
        _assert(
            checks,
            "best_rmse_threshold",
            score_f <= EXPECTED_BEST_RMSE_MAX,
            f"best_score={best_score} (max {EXPECTED_BEST_RMSE_MAX})",
        )

        model_results = job.get("model_results") or []
        got_models = {
            (mr.get("model") if isinstance(mr, dict) else None) for mr in model_results
        }
        for m in models:
            _assert(checks, f"trained_{m}", m in got_models, f"got={sorted(got_models - {None})}")

        # Per-model RMSE matches fixture table
        for mr in model_results:
            if not isinstance(mr, dict):
                continue
            name = mr.get("model")
            if name not in EXPECTED_MODEL_RMSE:
                continue
            rmse = (mr.get("metrics") or {}).get("rmse")
            expected = EXPECTED_MODEL_RMSE[name]
            try:
                ok_rmse = abs(float(rmse) - expected) < 1e-6
            except (TypeError, ValueError):
                ok_rmse = False
            _assert(
                checks,
                f"rmse_{name}",
                ok_rmse,
                f"rmse={rmse} expected={expected}",
            )

        extra["job"] = job

        # 6) Get job info
        if job_id:
            info_job = _http_json("GET", f"{base_url}/get-job-info?id={job_id}")
            _assert(
                checks,
                "job_info_best_model",
                info_job.get("best_model") == EXPECTED_BEST_MODEL,
                f"best={info_job.get('best_model')}",
            )
            extra["job_info"] = info_job

        # 7) List jobs for user
        jobs_resp = _http_json(
            "POST",
            f"{base_url}/get-list-job-by-userId?user_id=student_ci",
        )
        jobs = jobs_resp.get("jobs") if isinstance(jobs_resp, dict) else jobs_resp
        if not isinstance(jobs, list):
            jobs = []
        _assert(checks, "list_jobs_nonempty", len(jobs) >= 1, f"n_jobs={len(jobs)}")

    except Exception as exc:
        _assert(checks, "mock_api_exception", False, str(exc))
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    ok = all(c.ok for c in checks)
    return LayerReport(
        layer="mock-api",
        ok=ok,
        checks=checks,
        elapsed_seconds=round(time.time() - t0, 3),
        extra=extra,
    )


# ── Layer: agent (optional live) ─────────────────────────


async def run_agent_layer(*, user_id: str, models: Sequence[str]) -> LayerReport:
    print("\n═══ Layer: agent (live DeerFlow graph) ═══")
    from hagent.agent.graph import run_agent

    checks: List[CheckResult] = []
    t0 = time.time()
    models_s = ", ".join(models)
    messages = [
        "Hiển thị danh sách dataset của tôi",
        "Cho tôi xem thông tin chi tiết dataset Student Performance",
        (
            f"Huấn luyện dataset Student Performance với 3 thuật toán: {models_s}. "
            f"Target là cột G3, problem_type là regression, dataset_id là {STUDENT_DATASET_ID}."
        ),
        "So sánh kết quả training vừa xong. Model nào tốt nhất?",
    ]
    steps: List[Dict[str, Any]] = []

    try:
        for i, msg in enumerate(messages, 1):
            print(f"\n  Step {i}/{len(messages)}: {msg[:90]}...")
            start = time.time()
            result = await run_agent(message=msg, user_id=user_id)
            elapsed = time.time() - start
            response = str(result.get("response") or "")
            n_tools = len(result.get("tool_outputs") or [])
            steps.append(
                {
                    "step": i,
                    "message": msg,
                    "response_preview": response[:400],
                    "tool_calls": n_tools,
                    "route": result.get("route"),
                    "elapsed": round(elapsed, 2),
                    "tool_outputs": result.get("tool_outputs"),
                }
            )
            _assert(
                checks,
                f"agent_step_{i}_responded",
                bool(response) or n_tools > 0,
                f"elapsed={elapsed:.1f}s tools={n_tools} route={result.get('route')}",
            )

        # Training step should have called tools / mentioned models
        train_step = steps[2] if len(steps) >= 3 else {}
        outs = train_step.get("tool_outputs") or []
        payload_blob = json.dumps(outs, default=str)
        _assert(
            checks,
            "agent_train_has_tool_or_signal",
            train_step.get("tool_calls", 0) > 0
            or "RandomForest" in payload_blob
            or "XGB" in payload_blob
            or "best" in payload_blob.lower(),
            f"tools={train_step.get('tool_calls')}",
        )
    except Exception as exc:
        _assert(checks, "agent_exception", False, str(exc))

    ok = all(c.ok for c in checks)
    return LayerReport(
        layer="agent",
        ok=ok,
        checks=checks,
        elapsed_seconds=round(time.time() - t0, 3),
        extra={"steps": steps},
    )


# ── CLI ──────────────────────────────────────────────────


def _parse_layers(raw: str) -> List[str]:
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    if "all" in parts:
        return ["harness", "mock-api", "agent"]
    return parts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Student Performance realistic CI/CD E2E"
    )
    parser.add_argument(
        "--layers",
        default="harness,mock-api",
        help="Comma-separated: harness,mock-api,agent,all (default: harness,mock-api)",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("HAUTOML_BASE_URL", "http://127.0.0.1:8585"),
        help="Mock HAutoML base URL",
    )
    parser.add_argument("--port", type=int, default=8585, help="Mock server port")
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Do not spawn mock_hautoml_server (use existing base-url)",
    )
    parser.add_argument(
        "--models",
        default=",".join(STUDENT_MODELS),
        help="Comma-separated training models",
    )
    parser.add_argument("--user-id", default="student_ci")
    parser.add_argument("--json", dest="json_path", default=None)
    parser.add_argument("--md", dest="md_path", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    layers = _parse_layers(args.layers)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    reports: List[LayerReport] = []

    print("=" * 64)
    print("  Student Performance — CI/CD E2E")
    print("=" * 64)
    print(f"  Layers: {layers}")
    print(f"  Models: {models}")
    print(f"  Dataset: {STUDENT_DATASET_ID} target={STUDENT_TARGET}")

    if "harness" in layers:
        reports.append(asyncio.run(run_harness_layer()))

    if "mock-api" in layers:
        reports.append(
            run_mock_api_layer(
                base_url=args.base_url.rstrip("/"),
                models=models,
                start_server=not args.no_start_server,
                port=args.port,
            )
        )

    if "agent" in layers:
        reports.append(
            asyncio.run(run_agent_layer(user_id=args.user_id, models=models))
        )

    overall_ok = all(r.ok for r in reports) if reports else False
    summary = {
        "dataset": STUDENT_DATASET_ID,
        "target": STUDENT_TARGET,
        "models": models,
        "expected_best_model": EXPECTED_BEST_MODEL,
        "layers": [r.to_dict() for r in reports],
        "ok": overall_ok,
        "n_layers": len(reports),
        "n_failed_layers": sum(1 for r in reports if not r.ok),
        "n_checks": sum(len(r.checks) for r in reports),
        "n_failed_checks": sum(
            1 for r in reports for c in r.checks if not c.ok
        ),
    }

    # Markdown summary
    md_lines = [
        "# Student Performance E2E",
        "",
        f"**Overall:** {'PASS ✓' if overall_ok else 'FAIL ✗'}",
        f"**Dataset:** `{STUDENT_DATASET_ID}` · target `{STUDENT_TARGET}` · "
        f"shape {STUDENT_N_ROWS}×{STUDENT_N_COLS}",
        f"**Models:** {', '.join(models)} · expected best: `{EXPECTED_BEST_MODEL}`",
        "",
        "| Layer | Status | Checks | Time |",
        "|---|---|---:|---:|",
    ]
    for r in reports:
        failed = sum(1 for c in r.checks if not c.ok)
        md_lines.append(
            f"| {r.layer} | {'OK' if r.ok else 'FAIL'} | "
            f"{len(r.checks) - failed}/{len(r.checks)} | {r.elapsed_seconds}s |"
        )
    md_lines.append("")
    for r in reports:
        md_lines.append(f"## {r.layer}")
        for c in r.checks:
            mark = "✓" if c.ok else "✗"
            md_lines.append(f"- {mark} **{c.name}**: {c.detail}")
        md_lines.append("")
    md = "\n".join(md_lines)

    print("\n" + md)
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop huge nested harness report from JSON if present
        slim = dict(summary)
        for layer in slim.get("layers") or []:
            extra = layer.get("extra") or {}
            if "report" in extra:
                extra = {
                    k: v
                    for k, v in extra.items()
                    if k != "report"
                }
                extra["harness_n"] = (layer.get("extra") or {}).get("n")
                extra["harness_n_failed"] = (layer.get("extra") or {}).get(
                    "n_failed"
                )
                layer["extra"] = extra
        path.write_text(
            json.dumps(slim, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"📁 JSON → {path}")
    if args.md_path:
        path = Path(args.md_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"📁 Markdown → {path}")

    print(
        f"\n{'=' * 64}\n"
        f"  Result: {'PASS' if overall_ok else 'FAIL'} "
        f"({summary['n_failed_checks']} failed checks / {summary['n_checks']})\n"
        f"{'=' * 64}"
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
