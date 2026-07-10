"""Harness reporting (JSON + Markdown)."""

from __future__ import annotations

from typing import Any, Dict, List

from hagent.agent.harness.schema import AgentRunResult


def summarize_results(results: List[AgentRunResult]) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[AgentRunResult]] = {}
    for r in results:
        key = f"{r.layer}:{r.mode}"
        buckets.setdefault(key, []).append(r)
    summaries = []
    for key, rows in sorted(buckets.items()):
        n = len(rows)
        if not n:
            continue
        elapsed = sorted(r.elapsed_seconds for r in rows)
        p50 = elapsed[n // 2]
        p95 = elapsed[min(n - 1, int(n * 0.95))]
        summaries.append(
            {
                "key": key,
                "layer": rows[0].layer,
                "mode": rows[0].mode,
                "n": n,
                "success_rate": sum(1 for r in rows if r.success) / n,
                "avg_elapsed": sum(r.elapsed_seconds for r in rows) / n,
                "p50_elapsed": p50,
                "p95_elapsed": p95,
                "avg_tools": sum(r.tools_called for r in rows) / n,
            }
        )
    return summaries


def build_report(results: List[AgentRunResult], **meta: Any) -> Dict[str, Any]:
    return {
        "results": [r.to_dict() for r in results],
        "summaries": summarize_results(results),
        "n": len(results),
        "n_failed": sum(1 for r in results if not r.success),
        **meta,
    }


def report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Agent Harness Report",
        "",
        f"Total runs: {report.get('n', 0)} | Failed: {report.get('n_failed', 0)}",
        "",
        "## Summary",
        "",
        "| Layer:Mode | N | Success | Avg s | p50 | p95 | Avg tools |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in report.get("summaries") or []:
        lines.append(
            f"| {s['key']} | {s['n']} | {s['success_rate']:.0%} | "
            f"{s['avg_elapsed']:.3f} | {s['p50_elapsed']:.3f} | "
            f"{s['p95_elapsed']:.3f} | {s['avg_tools']:.1f} |"
        )
    lines.extend(["", "## Details", ""])
    for r in report.get("results") or []:
        status = "OK" if r.get("success") else "FAIL"
        lines.append(
            f"- **{r.get('scenario_id')}** `{r.get('layer')}/{r.get('mode')}`: "
            f"{status} ({r.get('elapsed_seconds')}s, tools={r.get('tools_called')}) "
            f"— {', '.join(r.get('reasons') or [])}"
        )
    return "\n".join(lines) + "\n"
