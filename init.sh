#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEATURE_FILE="${ROOT_DIR}/feature_list.json"
PROGRESS_FILE="${ROOT_DIR}/claude-progress.md"

fail() {
  printf 'ERROR: %s\n' "$1" >&2
  exit 1
}

note() {
  printf '%s\n' "$1"
}

note "== Minimal Agent Harness: initialize =="

[[ -f "${ROOT_DIR}/AGENTS.md" ]] || fail "Thiếu AGENTS.md"
[[ -f "${FEATURE_FILE}" ]] || fail "Thiếu feature_list.json"
[[ -f "${PROGRESS_FILE}" ]] || fail "Thiếu claude-progress.md"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  fail "Cần Python 3 để kiểm tra feature_list.json"
fi

"${PYTHON_BIN}" - "${FEATURE_FILE}" <<'PY'
import json
import pathlib
import re
import sys

feature_path = pathlib.Path(sys.argv[1])

try:
    data = json.loads(feature_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"ERROR: feature_list.json không hợp lệ: {exc}")

errors = []
valid_statuses = {"backlog", "in_progress", "blocked", "done"}
wildcard_re = re.compile(r"[*?\[\]]")

policy = data.get("policy")
features = data.get("features")
current_task_id = data.get("current_task_id")

if not isinstance(policy, dict):
    errors.append("policy phải là object")
    policy = {}

if policy.get("wip_limit") != 1:
    errors.append("policy.wip_limit bắt buộc bằng 1")

control_files = policy.get("control_files")
if not isinstance(control_files, list) or not control_files:
    errors.append("policy.control_files phải là danh sách không rỗng")

protected_paths = policy.get("protected_paths")
if not isinstance(protected_paths, list) or not protected_paths:
    errors.append("policy.protected_paths phải là danh sách không rỗng")

if not isinstance(features, list):
    errors.append("features phải là array")
    features = []

task_ids = set()
in_progress = []

def validate_exact_path(path, context):
    if not isinstance(path, str) or not path.strip():
        errors.append(f"{context}: đường dẫn phải là chuỗi không rỗng")
        return
    normalized = path.replace("\\", "/")
    relative_path = pathlib.PurePosixPath(normalized)
    windows_path = pathlib.PureWindowsPath(normalized)
    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
    ):
        errors.append(f"{context}: không được dùng đường dẫn tuyệt đối: {path}")
        return
    if ".." in relative_path.parts:
        errors.append(f"{context}: không được chứa '..': {path}")
        return
    candidate = feature_path.parent.joinpath(*relative_path.parts)
    if wildcard_re.search(normalized) and not candidate.is_file():
        errors.append(f"{context}: whitelist không được chứa wildcard: {path}")
    if normalized.endswith("/") or candidate.is_dir():
        errors.append(f"{context}: whitelist phải trỏ tới file, không phải thư mục: {path}")

if isinstance(control_files, list):
    for index, path in enumerate(control_files):
        validate_exact_path(path, f"policy.control_files[{index}]")

for index, feature in enumerate(features):
    context = f"features[{index}]"
    if not isinstance(feature, dict):
        errors.append(f"{context} phải là object")
        continue

    task_id = feature.get("id")
    status = feature.get("status")
    allowed_files = feature.get("allowed_files")
    test_commands = feature.get("test_commands")
    acceptance_criteria = feature.get("acceptance_criteria")
    verification = feature.get("verification")

    if not isinstance(task_id, str) or not task_id.strip():
        errors.append(f"{context}.id phải là chuỗi không rỗng")
    elif task_id in task_ids:
        errors.append(f"Trùng task id: {task_id}")
    else:
        task_ids.add(task_id)

    if status not in valid_statuses:
        errors.append(f"{context}.status không hợp lệ: {status!r}")
    elif status == "in_progress":
        in_progress.append(task_id)

    if not isinstance(acceptance_criteria, list) or not acceptance_criteria:
        errors.append(f"{context}.acceptance_criteria phải là danh sách không rỗng")

    if not isinstance(allowed_files, list) or not allowed_files:
        errors.append(f"{context}.allowed_files phải là danh sách không rỗng")
    else:
        for path_index, path in enumerate(allowed_files):
            validate_exact_path(path, f"{context}.allowed_files[{path_index}]")

    if not isinstance(test_commands, list) or not test_commands:
        errors.append(f"{context}.test_commands phải là danh sách không rỗng")

    if not isinstance(verification, dict):
        errors.append(f"{context}.verification phải là object")
    elif status == "done":
        if verification.get("status") != "passed":
            errors.append(f"{task_id}: task done phải có verification.status='passed'")
        if not verification.get("tested_at"):
            errors.append(f"{task_id}: task done phải có verification.tested_at")
        commands_run = verification.get("commands_run")
        if not isinstance(commands_run, list) or not commands_run:
            errors.append(f"{task_id}: task done phải ghi commands_run")

if len(in_progress) > 1:
    errors.append(f"WIP vượt quá 1: {in_progress}")

if in_progress:
    if current_task_id != in_progress[0]:
        errors.append(
            "current_task_id phải trùng task in_progress duy nhất "
            f"({in_progress[0]!r})"
        )
elif current_task_id is not None:
    errors.append("current_task_id phải là null khi không có task in_progress")

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)

active = in_progress[0] if in_progress else "none"
print(f"OK: JSON hợp lệ; WIP={len(in_progress)}/1; active={active}")
PY

if git -C "${ROOT_DIR}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  note "Git working tree:"
  git -C "${ROOT_DIR}" status --short
else
  note "WARN: Thư mục gốc chưa phải Git working tree; không thể kiểm tra diff."
fi

note "Nhật ký gần nhất:"
tail -n 12 "${PROGRESS_FILE}"

note "OK: Harness đã sẵn sàng. Hãy tiếp tục đúng task in_progress duy nhất."
