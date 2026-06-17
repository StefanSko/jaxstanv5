"""Structural tests for constraint metadata staying backend-free."""

from __future__ import annotations

from pathlib import Path

from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval

PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_builtin_constraints_are_metadata_only() -> None:
    for cls in (Interval, Ordered, Positive, UnitInterval):
        assert not hasattr(cls, "transform")
        assert not hasattr(cls, "inverse_transform")
        assert not hasattr(cls, "log_abs_det_jacobian")


def test_constraint_modules_have_no_jax_backend_imports() -> None:
    source_root = PROJECT_ROOT / "src" / "jaxstanv5" / "constraints"
    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import jax" not in source
        assert "jax.numpy" not in source
        assert "_jax_lazy" not in source
