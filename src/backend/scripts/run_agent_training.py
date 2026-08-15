"""
HAgent — Agent Training Runner.

Script chạy multi-agent graph với Ollama để training dataset.
Dùng trong GitHub Actions hoặc local.

Usage:
  python scripts/run_agent_training.py \
    --message "Huấn luyện dataset student với 3 thuật toán RF, XGBoost, SVR" \
    --user-id ci_user
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure backend is in path
BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


async def run_training(
    message: str,
    user_id: str,
    *,
    world_model: dict | None = None,
) -> dict:
    """Chạy agent graph với message training."""
    from hagent.agent.orchestration.graph import run_agent

    print("=" * 60)
    print("  HAgent — Agent Training Runner")
    print("=" * 60)
    print(f"\n📨 Message: {message}")
    print(f"👤 User: {user_id}")
    print(f"🤖 Provider: {os.getenv('LLM_PROVIDER', 'unknown')}")
    print(f"📊 Model: {os.getenv('LLM_MODEL', os.getenv('OLLAMA_MODEL', 'unknown'))}")
    if world_model:
        print(
            f"🌍 World Model: datasets={len((world_model or {}).get('datasets') or {})} "
            f"jobs={len((world_model or {}).get('jobs') or {})}"
        )
    print()

    start = time.time()

    result = await run_agent(
        message=message,
        user_id=user_id,
        world_model=world_model,
    )

    elapsed = time.time() - start

    print("─" * 60)
    print(f"\n✅ Agent Response ({elapsed:.1f}s):\n")
    print(result.get("response", "No response"))

    if result.get("tool_outputs"):
        print(f"\n🔧 Tool Outputs ({len(result['tool_outputs'])} calls):")
        for i, to in enumerate(result["tool_outputs"], 1):
            tool_name = to.get("tool_name", "unknown")
            payload = to.get("payload", {})
            print(f"\n  [{i}] {tool_name}:")
            if isinstance(payload, dict):
                # Show key info
                if "best_model" in payload:
                    print(f"      🏆 Best Model: {payload['best_model']}")
                    print(f"      📊 Best Score: {payload.get('best_score', 'N/A')}")
                if "model_results" in payload:
                    print(f"      📋 Models trained: {len(payload['model_results'])}")
                    for mr in payload["model_results"]:
                        m_name = mr.get("model", "?")
                        m_metrics = mr.get("metrics", {})
                        print(f"         - {m_name}: {m_metrics}")
                if "datasets" in payload:
                    print(f"      📁 Datasets: {len(payload['datasets'])}")
                if "status" in payload:
                    print(f"      📌 Status: {payload['status']}")
            else:
                print(f"      {str(payload)[:200]}")

    # World Model / planning surface
    if (
        result.get("plan_status")
        or result.get("surprise")
        or result.get("campaign_status")
    ):
        print("\n🌍 World Model / planning:")
        if result.get("plan_status"):
            print(f"      plan_status: {result.get('plan_status')}")
        if result.get("campaign_status"):
            print(f"      campaign_status: {result.get('campaign_status')}")
        if result.get("hierarchy_status"):
            print(f"      hierarchy_status: {result.get('hierarchy_status')}")
        sur = result.get("surprise") or {}
        if sur:
            print(f"      surprise: level={sur.get('level')} value={sur.get('value')}")
        if result.get("evaluation"):
            ev = result["evaluation"]
            print(
                f"      best_job: {ev.get('best_job_id')} rec={ev.get('recommendation')}"
            )

    print(f"\n📌 Route: {result.get('route', 'N/A')}")
    print(f"⏱️  Time: {elapsed:.1f}s")
    print(f"🔌 Provider: {result.get('provider', 'N/A')}")
    print("─" * 60)

    return result


async def run_conversation(user_id: str) -> list[dict]:
    """Chạy chuỗi conversation giả lập end-user flow."""
    from hagent.agent.orchestration.graph import run_agent

    messages = [
        "Hiển thị danh sách dataset của tôi",
        "Cho tôi xem thông tin chi tiết dataset student",
        "Huấn luyện dataset student với 3 thuật toán: RandomForestRegressor, XGBRegressor, SVR. Target là cột G3, problem_type là regression.",
        "Cho tôi xem kết quả training vừa xong",
    ]

    results = []
    print("=" * 60)
    print("  HAgent — Full Training Conversation")
    print("=" * 60)

    for i, msg in enumerate(messages, 1):
        print(f"\n{'━' * 60}")
        print(f"  Step {i}/{len(messages)}: {msg[:80]}")
        print(f"{'━' * 60}")

        start = time.time()
        result = await run_agent(message=msg, user_id=user_id)
        elapsed = time.time() - start

        response = result.get("response", "No response")
        n_tools = len(result.get("tool_outputs", []))
        route = result.get("route", "N/A")

        print(f"\n  🤖 Response ({elapsed:.1f}s, route={route}, tools={n_tools}):")
        # Truncate long responses
        if len(response) > 500:
            print(f"  {response[:500]}...")
        else:
            print(f"  {response}")

        # Show key tool outputs
        for to in result.get("tool_outputs", []):
            payload = to.get("payload", {})
            if isinstance(payload, dict) and "best_model" in payload:
                print(f"\n  🏆 BEST MODEL: {payload['best_model']}")
                print(f"  📊 BEST SCORE: {payload.get('best_score', 'N/A')}")
                if "model_results" in payload:
                    print("  📋 ALL RESULTS:")
                    for mr in payload["model_results"]:
                        print(f"     - {mr['model']}: {mr.get('metrics', {})}")

        results.append(
            {
                "step": i,
                "message": msg,
                "response_length": len(response),
                "tool_calls": n_tools,
                "route": route,
                "elapsed": round(elapsed, 1),
            }
        )

    print(f"\n{'═' * 60}")
    print("  📊 SUMMARY")
    print(f"{'═' * 60}")
    total_time = sum(r["elapsed"] for r in results)
    total_tools = sum(r["tool_calls"] for r in results)
    print(f"  Steps: {len(results)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Total tool calls: {total_tools}")
    print(f"  Routes: {[r['route'] for r in results]}")
    print(f"{'═' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description="HAgent Agent Training Runner")
    parser.add_argument(
        "--message",
        "-m",
        type=str,
        default=None,
        help="Single message to send to agent",
    )
    parser.add_argument("--user-id", "-u", type=str, default="ci_user", help="User ID")
    parser.add_argument(
        "--conversation",
        "-c",
        action="store_true",
        help="Run full conversation flow instead of single message",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--seed-glass-wm",
        action="store_true",
        help="Seed a glass dataset World Model snapshot for train prompts",
    )
    args = parser.parse_args()

    world_model = None
    if args.seed_glass_wm:
        world_model = {
            "user_id": args.user_id,
            "datasets": {
                "ds_glass": {
                    "id": "ds_glass",
                    "name": "glass",
                    "n_rows": 214,
                    "n_cols": 10,
                    "features": [
                        "RI",
                        "Na",
                        "Mg",
                        "Al",
                        "Si",
                        "K",
                        "Ca",
                        "Ba",
                        "Fe",
                        "Type",
                    ],
                    "target": "Type",
                }
            },
            "jobs": {},
            "active_dataset_id": "ds_glass",
            "phase": "idle",
        }

    if args.conversation:
        results = asyncio.run(run_conversation(args.user_id))
    elif args.message:
        result = asyncio.run(
            run_training(args.message, args.user_id, world_model=world_model)
        )
        results = [result]
    else:
        # Default: human-style training command (English) for CI demos
        msg = (
            "Please train a model on dataset ds_glass. "
            "Target column is Type, problem type classification, metric f1."
        )
        result = asyncio.run(run_training(msg, args.user_id, world_model=world_model))
        results = [result]

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n📁 Results saved to: {args.output}")


if __name__ == "__main__":
    main()
