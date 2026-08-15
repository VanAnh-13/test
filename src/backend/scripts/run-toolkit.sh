#!/usr/bin/env bash
set -Eeuo pipefail

backend_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${backend_dir}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${backend_dir}"
exec python -m server.application
