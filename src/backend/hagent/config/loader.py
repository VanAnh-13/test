"""
Bộ tải cấu hình HAgent dùng để tải, gộp và kiểm tra cấu hình.

Thứ tự ưu tiên (cao → thấp):
  1. Biến môi trường luôn ghi đè tất cả nguồn khác.
  2. Các file YAML dạng module (config/defaults.yaml, config/llm.yaml,
     config/world_model.yaml, config/agents.yaml)
  3. File config/hagent.yaml nguyên khối để tương thích ngược.
  4. Giá trị mặc định tích hợp sẵn trong schema Pydantic.

Chiến lược gộp:
  - Các file YAML được gộp theo thứ tự: defaults → world_model → llm → agents.
  - Nếu cùng khóa có ở cả file module và hagent.yaml, file module được ưu tiên.
  - Gộp sâu các dict theo đệ quy; list được thay thế hoàn toàn.

Tương thích ngược:
  - `load_raw_config()` trả dict như trước nên code sử dụng không cần sửa.
  - `load_typed_config()` trả `HAgentConfig` đã được Pydantic kiểm tra.
  - `bridge/config.py` tiếp tục hoạt động không thay đổi

Lý do thiết kế:
  - `lru_cache` giống `bridge/config.py`: tải một lần và lưu đệm suốt vòng đời.
  - Pydantic kiểm tra khi tải, vì vậy lỗi schema được phát hiện ngay khi import
    thay vì khi truy cập trường.
  - `_resolve_env_vars` và `_deep_resolve` giữ nguyên để tương thích với
    cú pháp ${VAR} và ${VAR:-default} hiện có trong config/hagent.yaml
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from hagent.config.schema import HAgentConfig

# ── Đường dẫn ─────────────────────────────────────────────────────────────────

# Thư mục chứa các file cấu hình YAML dạng module.
_CONFIG_DIR = Path(__file__).parent

# Thứ tự gộp các file module; kết quả cuối là một dict.
_MODULAR_FILES = [
    _CONFIG_DIR / "defaults.yaml",
    _CONFIG_DIR / "world_model.yaml",
    _CONFIG_DIR / "llm.yaml",
    _CONFIG_DIR / "agents.yaml",
]

# Tìm file hagent.yaml nguyên khối, ưu tiên vị trí thuộc package config.
_MONOLITH_SEARCH_PATHS = [
    _CONFIG_DIR / "hagent.yaml",  # hagent/config/hagent.yaml
    _CONFIG_DIR.parent / "hagent.yaml",  # vị trí cũ hagent/hagent.yaml
    _CONFIG_DIR.parent.parent / "hagent.yaml",  # backend/hagent.yaml
    Path.home() / ".hagent" / "hagent.yaml",  # ~/.hagent/hagent.yaml
]


# ── Phân giải biến môi trường ────────────────────────────────────────────────


def _resolve_env_vars(value: Any) -> Any:
    """
    Thay thế chuỗi dạng ${VAR_NAME} bằng giá trị biến môi trường.
    Hỗ trợ cả giá trị mặc định: ${VAR_NAME:-default}.
    """
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        inner = value[2:-1]
        if ":-" in inner:
            var_name, default = inner.split(":-", 1)
            return os.getenv(var_name, default)
        return os.getenv(inner, "")
    return value


def _deep_resolve(data: Any) -> Any:
    """Đệ quy thay thế tất cả biến môi trường trong cấu trúc dữ liệu."""
    if isinstance(data, dict):
        return {k: _deep_resolve(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_deep_resolve(item) for item in data]
    else:
        return _resolve_env_vars(data)


# ── Gộp sâu ──────────────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Gộp sâu hai dict. Dict con được gộp theo đệ quy;
    giá trị vô hướng và list được thay thế bởi giá trị ghi đè.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


# ── Load từng nguồn ────────────────────────────────────────────────────────────


def _load_yaml_file(path: Path) -> dict:
    """Tải một file YAML và trả về dict rỗng nếu file không tồn tại."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw if isinstance(raw, dict) else {}


def _load_monolith() -> dict:
    """
    Tải file hagent.yaml nguyên khối bằng biến HAGENT_CONFIG hoặc tự động tìm.

    Trả về dict rỗng nếu không tìm thấy file khi chỉ dùng các file module.
    Nếu HAGENT_CONFIG trỏ tới file không tồn tại thì vẫn phát sinh lỗi.
    """
    env_path = os.getenv("HAGENT_CONFIG")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return _load_yaml_file(p)
        raise FileNotFoundError(f"HAGENT_CONFIG trỏ tới file không tồn tại: {env_path}")

    for p in _MONOLITH_SEARCH_PATHS:
        if p.exists():
            return _load_yaml_file(p)

    return {}


def _load_modular() -> dict:
    """
    Tải và gộp các file YAML dạng module theo thứ tự.
    File sau ghi đè file trước bằng phép gộp sâu.
    """
    merged: dict = {}
    for path in _MODULAR_FILES:
        partial = _load_yaml_file(path)
        if partial:
            merged = _deep_merge(merged, partial)
    return merged


# ── API công khai ────────────────────────────────────────────────────────────


@lru_cache
def load_raw_config() -> dict:
    """
    Tải, gộp và lưu đệm toàn bộ cấu hình thành một dict thuần.

    Thứ tự merge:
      1. File hagent.yaml nguyên khối nếu có làm lớp cơ sở.
      2. Các file YAML dạng module ghi đè lên file nguyên khối.
      3. Phân giải biến môi trường sau khi gộp.

    Bộ nhớ đệm được xóa tự động khi khởi động lại; trong test dùng
    ``load_raw_config.cache_clear()`` hoặc mock.

    Giá trị trả về:
        Dict đã phân giải biến môi trường, tương thích với ``load_config()``
        của bridge/config.py nên code sử dụng không cần sửa.
    """
    monolith = _load_monolith()
    modular = _load_modular()

    # Các file dạng module được ưu tiên hơn file nguyên khối.
    merged = _deep_merge(monolith, modular)

    return _deep_resolve(merged)


@lru_cache
def load_config() -> HAgentConfig:
    """
    Tải, gộp và kiểm tra cấu hình thành đối tượng có kiểu.

    Ngoại lệ:
        pydantic.ValidationError: Nếu config không hợp lệ.

    Giá trị trả về:
        HAgentConfig là đối tượng cấu hình có kiểu và đầy đủ giá trị mặc định.
    """
    raw = load_raw_config()
    return HAgentConfig.model_validate(raw)


# Giữ cả thuộc tính cache_clear/cache_info của API cũ.
load_typed_config = load_config


def clear_cache() -> None:
    """Xóa toàn bộ bộ nhớ đệm của bộ tải để phục vụ kiểm thử hoặc nạp lại nóng."""
    load_raw_config.cache_clear()
    load_config.cache_clear()
