"""Regression tests cho automl.pipeline.preprocessor.

Bảo đảm loader model KHÔNG dùng eval() và chỉ chấp nhận các model đã đăng ký
trong registry an toàn (_MODEL_CLASSES).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from automl.pipeline.errors import UnknownModelError
from automl.pipeline.preprocessor import _MODEL_CLASSES, get_model


def _write_classification_yaml(base_dir: Path, model_name: str) -> None:
    content = textwrap.dedent(
        f"""
        metric_list:
          - accuracy
        Classification_models:
          0:
            model: {model_name}
            params: []
        """
    ).strip()
    (base_dir / "classification.yml").write_text(content, encoding="utf-8")


def test_get_model_loads_registered_models_from_default_config() -> None:
    models, metric_list = get_model()

    assert models, "Phải nạp được ít nhất một model từ cấu hình hệ thống"
    assert isinstance(metric_list, list) and metric_list
    for entry in models.values():
        assert type(entry["model"]) in _MODEL_CLASSES.values()


def test_get_model_accepts_registered_model_name(tmp_path: Path) -> None:
    _write_classification_yaml(tmp_path, "RandomForestClassifier")

    models, _ = get_model(base_dir=str(tmp_path))

    assert type(models[0]["model"]).__name__ == "RandomForestClassifier"


def test_get_model_rejects_unregistered_model_name(tmp_path: Path) -> None:
    _write_classification_yaml(tmp_path, "NotARealModel")

    with pytest.raises(UnknownModelError) as exc_info:
        get_model(base_dir=str(tmp_path))

    assert exc_info.value.model_name == "NotARealModel"


def test_get_model_does_not_execute_arbitrary_code(tmp_path: Path) -> None:
    # Nếu còn dùng eval(), tên này sẽ raise NameError khác thay vì UnknownModelError.
    _write_classification_yaml(tmp_path, "__import__('os').getcwd()")

    with pytest.raises(UnknownModelError):
        get_model(base_dir=str(tmp_path))
