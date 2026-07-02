"""Structural tests for distribution metadata staying backend-free."""

from __future__ import annotations

from pathlib import Path

from jaxstanv5.distributions import (
    Bernoulli,
    Beta,
    BetaBinomial,
    Binomial,
    Exponential,
    HalfNormal,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    OrderedLogistic,
    Poisson,
    StudentT,
    Truncated,
    Uniform,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def test_builtin_distributions_are_metadata_only() -> None:
    for cls in (
        Bernoulli,
        Beta,
        BetaBinomial,
        Binomial,
        Exponential,
        HalfNormal,
        MultivariateNormal,
        NegativeBinomial,
        Normal,
        OrderedLogistic,
        Poisson,
        StudentT,
        Truncated,
        Uniform,
    ):
        assert not hasattr(cls, "log_prob")
        assert not hasattr(cls, "sample")
        assert not hasattr(cls, "batch_shape")
        assert not hasattr(cls, "event_shape")


def test_distribution_modules_have_no_jax_backend_imports() -> None:
    source_root = PROJECT_ROOT / "src" / "jaxstanv5" / "distributions"
    for path in source_root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "import jax" not in source
        assert "jax.numpy" not in source
        assert "_jax_lazy" not in source
