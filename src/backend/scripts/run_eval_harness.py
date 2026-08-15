#!/usr/bin/env python3
"""
Deprecated alias → scripts/run_agent_harness.py (offline layer).

Prefer:
  python scripts/run_agent_harness.py --layer offline,graph
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import warnings
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


async def _main() -> int:
    warnings.warn(
        "run_eval_harness.py is deprecated; use run_agent_harness.py",
        DeprecationWarning,
        stacklevel=1,
    )
    from hagent.agent.harness import report_markdown, run_harness_suite

    parser = argparse.ArgumentParser(
        description="[deprecated] Offline eval → use run_agent_harness.py"
    )
    parser.add_argument("--modes", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--ids", type=str, default=None)
    parser.add_argument("--json", dest="json_path", type=str, default=None)
    parser.add_argument("--md", dest="md_path", type=str, default=None)
    args = parser.parse_args()

    modes = (
        [x.strip() for x in args.modes.split(",") if x.strip()] if args.modes else None
    )
    tags = [x.strip() for x in args.tags.split(",") if x.strip()] if args.tags else None
    ids = [x.strip() for x in args.ids.split(",") if x.strip()] if args.ids else None

    report = await run_harness_suite(
        layers=["offline"],
        offline_modes=modes,
        tags=tags,
        scenario_ids=ids,
    )
    md = report_markdown(report)
    print(md)
    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if args.md_path:
        Path(args.md_path).write_text(md, encoding="utf-8")
    return 1 if report.get("n_failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
