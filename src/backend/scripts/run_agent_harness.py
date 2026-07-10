#!/usr/bin/env python3
"""
Agent Harness CLI — offline / graph / api layers for DeerFlow-AutoML HAgent.

Usage:
  cd src/backend
  python scripts/run_agent_harness.py
  python scripts/run_agent_harness.py --layer offline,graph --tags smoke
  python scripts/run_agent_harness.py --layer graph --ids smoke_train_glass --json /tmp/h.json
  python scripts/run_agent_harness.py --layer api --base-url http://localhost:5360 --token "$JWT"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


async def _main() -> int:
    from hagent.agent.harness import report_markdown, run_harness_suite

    parser = argparse.ArgumentParser(description="DeerFlow-AutoML Agent Harness")
    parser.add_argument(
        "--layer",
        type=str,
        default="offline,graph",
        help="Comma-separated: offline,graph,api,all",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Offline modes: single_shot,plan_executor,campaign,hierarchical",
    )
    parser.add_argument("--tags", type=str, default=None, help="Filter tags")
    parser.add_argument("--ids", type=str, default=None, help="Filter scenario ids")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--token", type=str, default=None, help="JWT for API layer")
    parser.add_argument(
        "--require-live",
        action="store_true",
        help="Fail API layer if stack is down (default: soft-skip)",
    )
    parser.add_argument("--json", dest="json_path", type=str, default=None)
    parser.add_argument("--md", dest="md_path", type=str, default=None)
    args = parser.parse_args()

    layers = [x.strip() for x in args.layer.split(",") if x.strip()]
    modes = (
        [x.strip() for x in args.modes.split(",") if x.strip()]
        if args.modes
        else None
    )
    tags = [x.strip() for x in args.tags.split(",") if x.strip()] if args.tags else None
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None

    report = await run_harness_suite(
        layers=layers,
        offline_modes=modes,
        tags=tags,
        scenario_ids=ids,
        api_base_url=args.base_url,
        api_token=args.token,
        require_live=args.require_live,
    )
    md = report_markdown(report)
    print(md)

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote JSON → {args.json_path}", file=sys.stderr)
    if args.md_path:
        Path(args.md_path).write_text(md, encoding="utf-8")
        print(f"Wrote Markdown → {args.md_path}", file=sys.stderr)

    return 1 if report.get("n_failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
