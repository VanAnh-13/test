#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEFAULT_WORKERS=1

usage() {
    echo "Cách dùng: $0 [-n <số_worker>] [-b]"
    echo "  -n  Số worker cần chạy; stack hiện tại hỗ trợ đúng 1 worker."
    echo "  -b  Buộc Docker Compose build lại image."
}

num_workers="${DEFAULT_WORKERS}"
force_build=false

while getopts ":n:bh" option; do
    case "${option}" in
        n)
            num_workers="${OPTARG}"
            ;;
        b)
            force_build=true
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "Thiếu giá trị cho tùy chọn -${OPTARG}." >&2
            usage >&2
            exit 2
            ;;
        \?)
            echo "Tùy chọn không hợp lệ: -${OPTARG}." >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! "${num_workers}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Số worker phải là số nguyên dương." >&2
    exit 2
fi
if [[ "${num_workers}" -ne "${DEFAULT_WORKERS}" ]]; then
    echo "Stack hiện tại chỉ hỗ trợ đúng ${DEFAULT_WORKERS} worker." >&2
    exit 2
fi

backend_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${backend_dir}"

if [[ "${force_build}" == true ]]; then
    exec docker compose --profile worker up --build -d hautoml_woker_1
fi
exec docker compose --profile worker up -d hautoml_woker_1
