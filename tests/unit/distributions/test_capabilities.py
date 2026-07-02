"""Tests for backend-neutral distribution capability metadata."""

from __future__ import annotations

import pytest

from jaxstanv5._backends.jax.distributions import is_inverse_cdf
from jaxstanv5.distributions import Exponential, HalfNormal, Normal, StudentT, Truncated, Uniform
from jaxstanv5.distributions._capabilities import has_scalar_inverse_cdf
from jaxstanv5.distributions.core import Distribution


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [
        (Normal(0.0, 1.0), True),
        (HalfNormal(1.0), True),
        (Exponential(1.0), True),
        (Uniform(0.0, 1.0), True),
        (StudentT(4.0, 0.0, 1.0), False),
        (Truncated(Normal(0.0, 1.0), lower=0.0), True),
    ],
)
def test_scalar_inverse_cdf_capability_matches_backend_dispatch(
    distribution: Distribution,
    expected: bool,
) -> None:
    assert has_scalar_inverse_cdf(distribution) is expected
    assert is_inverse_cdf(distribution) is expected
