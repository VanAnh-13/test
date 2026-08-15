"""Regression cho quy tắc không đặt source entrypoint ở backend root."""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "run-hautoml-nano-docker.sh",
    "run-hautoml-toolkit-docker.sh",
    "run-toolkit.sh",
    "run-worker.sh",
)


def test_shell_entrypoints_are_owned_by_scripts_package() -> None:
    scripts_dir = BACKEND_ROOT / "scripts"

    assert (scripts_dir / "__init__.py").is_file()
    for name in SCRIPT_NAMES:
        assert not (BACKEND_ROOT / name).exists()
        assert (scripts_dir / name).is_file()


def test_shell_entrypoints_do_not_reference_removed_dockerfiles() -> None:
    scripts_dir = BACKEND_ROOT / "scripts"
    combined_source = "\n".join(
        (scripts_dir / name).read_text(encoding="utf-8") for name in SCRIPT_NAMES
    )

    assert "hautoml.nano.dockerfile" not in combined_source
    assert "worker.dockerfile" not in combined_source
    assert "worker.docker-compose.yaml" not in combined_source
    assert "docker-compose " not in combined_source
    assert "server.application" in combined_source
