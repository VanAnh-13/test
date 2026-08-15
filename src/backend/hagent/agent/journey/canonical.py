"""Các hàm serialization chuẩn dùng chung cho những toán tử Journey."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


def canonical_mapping_hash(value: Mapping[str, Any]) -> str:
    """Trả về mã băm SHA-256 UTF-8 ổn định của một mapping JSON."""
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
