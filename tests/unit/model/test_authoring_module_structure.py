"""Structural tests for authoring modules staying backend-free."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def test_model_decorator_has_no_jax_backend_imports() -> None:
    source = (PROJECT_ROOT / "src" / "jaxstanv5" / "model" / "decorator.py").read_text(
        encoding="utf-8"
    )

    assert "import jax" not in source
    assert "jax.numpy" not in source
    assert "_jax_lazy" not in source
