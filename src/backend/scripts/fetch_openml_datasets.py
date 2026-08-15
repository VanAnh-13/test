#!/usr/bin/env python3
"""
Tải bộ dataset OpenML cho benchmark — dùng sklearn.datasets.fetch_openml
(sklearn đã có sẵn trong requirements, không thêm dependency).

Usage:
  cd src/backend
  python scripts/fetch_openml_datasets.py --list
  python scripts/fetch_openml_datasets.py                      # tải cả 8 bộ
  python scripts/fetch_openml_datasets.py --datasets credit-g,diabetes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parent.parent
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# data_id cố định trên OpenML — không dùng name+version để tránh trôi phiên bản
DATASETS: dict[str, dict[str, Any]] = {
    "credit-g": {"data_id": 31, "problem_type": "classification"},
    "diabetes": {"data_id": 37, "problem_type": "classification"},
    "vehicle": {"data_id": 54, "problem_type": "classification"},
    "blood-transfusion": {"data_id": 1464, "problem_type": "classification"},
    "banknote": {"data_id": 1462, "problem_type": "classification"},
    "kc1": {"data_id": 1067, "problem_type": "classification"},
    "phoneme": {"data_id": 1489, "problem_type": "classification"},
    "wine-quality-red": {"data_id": 40691, "problem_type": "classification"},
}

DEFAULT_OUT = BACKEND / "assets" / "openml"


def normalize_frame(df, target_col: str):
    """Chuẩn hóa DataFrame: cột target đổi tên 'target', bỏ hàng thiếu target."""
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not in frame")
    out = df.rename(columns={target_col: "target"})
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    return out


def fetch_one(name: str, out_dir: Path) -> dict[str, Any]:
    """Tải một dataset, ghi CSV, trả manifest entry."""
    from sklearn.datasets import fetch_openml

    spec = DATASETS[name]
    bunch = fetch_openml(data_id=spec["data_id"], as_frame=True, parser="auto")
    df = bunch.frame
    target_col = (
        bunch.target_names[0] if getattr(bunch, "target_names", None) else "class"
    )
    if target_col not in df.columns:
        # fallback: cột cuối là target
        target_col = df.columns[-1]
    df = normalize_frame(df, target_col)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    entry = {
        "name": name,
        "data_id": spec["data_id"],
        "problem_type": spec["problem_type"],
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target": "target",
        "file": csv_path.name,
    }
    print(f"  {name}: {entry['n_rows']} rows x {entry['n_cols']} cols -> {csv_path}")
    return entry


def _main() -> int:
    parser = argparse.ArgumentParser(description="Fetch OpenML benchmark datasets")
    parser.add_argument(
        "--list", action="store_true", help="Liệt kê registry rồi thoát"
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help="Danh sách tên, phân tách dấu phẩy",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Thư mục output")
    args = parser.parse_args()

    if args.list:
        for name, spec in DATASETS.items():
            print(f"{name:<20} data_id={spec['data_id']:<6} {spec['problem_type']}")
        return 0

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"Unknown datasets: {unknown}. Available: {', '.join(DATASETS)}")

    out_dir = Path(args.out)
    manifest = []
    failed = []
    for name in names:
        try:
            manifest.append(fetch_one(name, out_dir))
        except Exception as exc:
            failed.append((name, str(exc)))
            print(f"  {name}: FAILED — {exc}", file=sys.stderr)

    if manifest:
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Manifest: {manifest_path} ({len(manifest)} datasets)")
    if failed:
        print(
            f"{len(failed)} dataset(s) failed: {[n for n, _ in failed]}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
