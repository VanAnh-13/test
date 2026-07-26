"""
HAgent — Full System Docker E2E Test.

Flow:
  1. Register + login (JWT)
  2. Upload Online Shoppers Intention dataset
  3. Send train prompt via HAgent Bridge (agent-run)
  4. Poll conversation / jobs until training is accepted or jobs exist
  5. Assert ≥1 job for the user

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

# URLs
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5370").rstrip("/")
HAGENT_URL = os.environ.get("HAGENT_URL", "http://localhost:5360").rstrip("/")
DATASET_PATH = os.environ.get(
    "HAGENT_TEST_DATASET",
    str(
        Path(__file__).parent.parent
        / "assets"
        / "online_shoppers"
        / "online_shoppers_intention.csv"
    ),
)

# Polling
MAX_RETRIES = int(os.environ.get("E2E_MAX_RETRIES", "36"))  # ~6 min default
RETRY_INTERVAL = float(os.environ.get("E2E_RETRY_INTERVAL", "10"))


def _jwt_user_id(token: str) -> str:
    token_parts = token.split(".")
    payload_b64 = token_parts[1]
    payload_b64 += "=" * (4 - len(payload_b64) % 4)
    payload = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
    return str(payload.get("sub") or "")


def _agent_response_is_error(msg: str) -> bool:
    lower = (msg or "").lower()
    markers = (
        "đang gặp lỗi",
        "gặp lỗi khi xử lý",
        "mất kết nối",
        "lỗi hagent",
        "provider\": \"error",
        "runtime error",
    )
    return any(m in lower for m in markers) or (
        msg.strip().startswith("⚠️") and "job" not in lower and "train" not in lower
    )


def _messages_indicate_success(messages: list) -> bool:
    success_markers = (
        "job training đã hoàn tất",
        "✅ job training",
        "best_model",
        "best model",
        "job_id",
        "đã bắt đầu training",
        "training đã",
        "campaign",
        "hierarchy",
        "start_training",
    )
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "").lower()
        if any(m in content for m in success_markers):
            return True
    return False


async def _list_jobs(client: httpx.AsyncClient, token: str, user_id: str) -> list:
    r = await client.post(
        f"{BASE_URL}/get-list-job-by-userId?user_id={user_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    if r.status_code != 200:
        print(f"  ! List jobs HTTP {r.status_code}: {r.text[:200]}")
        return []
    data = r.json()
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        jobs = data.get("jobs") or data.get("data") or []
        return jobs if isinstance(jobs, list) else []
    return []


async def test_e2e_docker() -> int:
    print("=" * 60)
    print("  🚀 HAgent — Full System Docker E2E Test")
    print("=" * 60)
    print(f"  HAutoML Toolkit URL: {BASE_URL}")
    print(f"  HAgent Bridge URL:   {HAGENT_URL}")
    print(f"  Dataset path:        {DATASET_PATH}")
    print("=" * 60)

    username = f"ci_test_{int(time.time())}"
    email = f"{username}@example.com"
    password = "password123"

    # 1. Register
    print("\n[1/6] Registering test user...")
    async with httpx.AsyncClient(timeout=30) as client:
        signup_payload = {
            "username": username,
            "email": email,
            "gender": "male",
            "date": "01/01/2026",
            "number": "0123456789",
            "fullName": "CI Test User",
            "password": password,
        }
        try:
            r = await client.post(f"{BASE_URL}/signup", json=signup_payload)
            if r.status_code == 200:
                print("  ✓ User registered successfully!")
            elif r.status_code == 409:
                print("  ! User already exists, proceeding to login...")
            else:
                print(f"  ✗ Signup failed (HTTP {r.status_code}): {r.text}")
                return 1
        except Exception as e:
            print(f"  ✗ Connection to master failed: {e}")
            return 1

    # 2. Login
    print("\n[2/6] Logging in to retrieve JWT token...")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{BASE_URL}/login",
            json={"username": username, "password": password},
        )
        if r.status_code != 200:
            print(f"  ✗ Login failed (HTTP {r.status_code}): {r.text}")
            return 1
        token = r.json()["access_token"]
        user_id = _jwt_user_id(token)
        print("  ✓ Login success!")
        print(f"  👤 User ID: {user_id}")

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Upload dataset
    print("\n[3/6] Uploading Online Shoppers Intention dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"  ✗ Dataset file not found at: {DATASET_PATH}")
        return 1

    async with httpx.AsyncClient(timeout=60) as client:
        with open(DATASET_PATH, "rb") as f:
            files = {
                "file_data": (
                    "online_shoppers_intention.csv",
                    f,
                    "text/csv",
                )
            }
            data = {
                "data_name": "online_shoppers_intention_ci",
                "data_type": "csv",
            }
            r = await client.post(
                f"{BASE_URL}/upload-dataset?user_id={user_id}",
                data=data,
                files=files,
                headers=headers,
            )
            if r.status_code != 200:
                print(f"  ✗ Dataset upload failed (HTTP {r.status_code}): {r.text}")
                return 1
            ds_result = r.json()
            dataset_id = ds_result.get("_id") or ds_result.get("id")
            print("  ✓ Dataset uploaded successfully!")
            print(f"    Dataset ID: {dataset_id}")

    # 4. Train via HAgent Bridge — prompt uses patterns goal_parser must catch
    print("\n[4/6] Sending training prompt to HAgent Bridge...")
    chat_prompt = (
        f"Hãy train một model classification trên dataset ID {dataset_id} "
        f"với target column là 'Revenue', dùng 3 thuật toán: "
        f"RandomForestClassifier, XGBClassifier, SVC. "
        f"Dùng metric là accuracy. Hãy cấu hình và bắt đầu training giúp tôi."
    )
    print(f'  Prompt: "{chat_prompt}"')

    conversation_id = None
    response_msg = ""
    async with httpx.AsyncClient(timeout=300) as client:
        # Preflight health
        try:
            hr = await client.get(f"{HAGENT_URL}/api/v1/chat/health")
            print(f"  Bridge health: HTTP {hr.status_code} {hr.text[:200]}")
        except Exception as exc:
            print(f"  ! Bridge health check failed: {exc}")

        r = await client.post(
            f"{HAGENT_URL}/api/v1/chat/",
            json={
                "message": chat_prompt,
                "conversation_id": None,
                "context": {
                    "dataset_id": dataset_id,
                    "target_column": "Revenue",
                    "problem_type": "classification",
                    "metric": "accuracy",
                    "models": [
                        "RandomForestClassifier",
                        "XGBClassifier",
                        "SVC",
                    ],
                },
            },
            headers=headers,
        )
        if r.status_code != 200:
            print(f"  ✗ Chat request failed (HTTP {r.status_code}): {r.text}")
            return 1

        chat_res = r.json()
        conversation_id = chat_res.get("conversation_id")
        response_msg = str(chat_res.get("message") or "")
        provider = chat_res.get("provider", "")
        print("  ✓ Prompt accepted by agent!")
        print(f"    Conversation ID: {conversation_id}")
        print(f"    Provider:         {provider}")
        print(f"    Agent Response:   {response_msg[:500]}")

        if _agent_response_is_error(response_msg) or provider == "error":
            print("  ✗ Agent returned an error response (see toolkit/bridge logs).")
            # Still poll jobs briefly in case training was side-effected
        else:
            print("  ✓ Agent response is not an error envelope")

    # 5. Poll conversation + jobs
    print("\n[5/6] Polling conversation history / jobs for training...")
    print("  (AutoML worker will pick up the task and train models...)")

    training_success = False
    jobs_found: list = []
    final_messages: list = []

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{attempt}/{MAX_RETRIES}] Polling messages + jobs...")
        async with httpx.AsyncClient(timeout=30) as client:
            if conversation_id:
                r = await client.get(
                    f"{HAGENT_URL}/api/v1/chat/conversation/{conversation_id}",
                    headers=headers,
                )
                if r.status_code == 200:
                    history = r.json()
                    messages = history.get("messages", [])
                    final_messages = messages
                    if _messages_indicate_success(messages):
                        print("\n  🎉 Training signal found in conversation history!")
                        for msg in messages:
                            if msg.get("role") == "assistant":
                                print(f"  {'-' * 60}")
                                print(msg.get("content", "")[:800])
                                print(f"  {'-' * 60}")
                        training_success = True

            jobs_found = await _list_jobs(client, token, user_id)
            if jobs_found:
                print(f"  ✓ Jobs visible: {len(jobs_found)}")
                training_success = True

        if training_success:
            break
        await asyncio.sleep(RETRY_INTERVAL)

    # 6. Final job listing
    print("\n[6/6] Checking training jobs status from AutoML Master...")
    async with httpx.AsyncClient(timeout=30) as client:
        jobs_found = await _list_jobs(client, token, user_id)
        print(f"  ✓ Found {len(jobs_found)} job(s) for user {user_id}:")
        for job in jobs_found[:10]:
            if not isinstance(job, dict):
                continue
            jid = job.get("_id") or job.get("id") or job.get("job_id")
            status = job.get("status")
            best_model = (
                job.get("best_model_name")
                or job.get("best_model")
                or "N/A"
            )
            best_score = job.get("best_score", "N/A")
            print(
                f"    - Job ID: {jid} | Status: {status} | "
                f"Best Model: {best_model} | Score: {best_score}"
            )

    # Hard success criterion: at least one training job in AutoML Master.
    # "Plan status: done" alone is NOT enough (agent may finish hierarchy without jobs).
    if jobs_found:
        print("\n" + "=" * 60)
        print("  ✅ DOCKER FULL SYSTEM E2E TEST PASSED SUCCESSFULLY!")
        print("     (Training job(s) created via HAgent agent path)")
        print("=" * 60)
        return 0

    print("\n" + "=" * 60)
    print("  ❌ DOCKER FULL SYSTEM E2E TEST FAILED OR TIMED OUT!")
    print(f"     Last agent response: {response_msg[:300]}")
    print(f"     Conversation messages: {len(final_messages)}")
    print(f"     Jobs found: {len(jobs_found)}")
    if "plan status: done" in response_msg.lower() and not jobs_found:
        print(
            "     Hint: agent completed planning but start_training did not "
            "persist a job — check toolkit logs for /v2/auto/jobs/training."
        )
    print("=" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(test_e2e_docker()))
