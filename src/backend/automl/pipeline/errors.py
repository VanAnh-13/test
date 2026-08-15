"""Ngoại lệ chuyên biệt cho AutoML Pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Exception cơ sở cho các lỗi trong AutoML pipeline."""


class UnknownModelError(PipelineError):
    """Tên model trong cấu hình không nằm trong registry model an toàn.

    Được ném thay cho ``eval()`` để tránh thực thi mã tùy ý từ tên model
    đọc trong file cấu hình YAML.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        super().__init__(
            f"Model không được hỗ trợ: '{model_name}'. "
            f"Hãy đăng ký class model trong registry _MODEL_CLASSES trước khi dùng."
        )
