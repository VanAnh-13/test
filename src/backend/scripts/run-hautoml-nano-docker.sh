#!/usr/bin/env bash
set -Eeuo pipefail

backend_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${backend_dir}"

exec docker compose --profile nano up --build -d hautoml_nano
