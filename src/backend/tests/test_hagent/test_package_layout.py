"""Kiểm tra cấu trúc package công khai của HAgent."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

HAGENT_ROOT = Path(__file__).parents[2] / "hagent"
LOOSE_ROOT_FILES = (
    "chat_router.py",
    "chat_store.py",
    "hagent.yaml",
    "logging.py",
    "run_models.py",
    "run_router.py",
)


def test_hagent_khong_con_file_roi_da_chuyen_vao_package():
    assert not [name for name in LOOSE_ROOT_FILES if (HAGENT_ROOT / name).exists()]
    assert (HAGENT_ROOT / "config" / "hagent.yaml").is_file()


def test_cac_package_moi_cung_cap_dung_boundary_runtime():
    chat_router = import_module("hagent.chat.router")
    chat_store = import_module("hagent.chat.store")
    run_models = import_module("hagent.run.models")
    run_router = import_module("hagent.run.router")
    logging_module = import_module("hagent.observability.logging")

    assert chat_router.router.prefix == "/api/v1/chat"
    assert callable(chat_store.add_message)
    assert run_models.StartRunRequest.model_fields["message"].is_required()
    assert run_router.router.prefix == "/api/v1/runs"
    assert callable(logging_module.configure_logging)
