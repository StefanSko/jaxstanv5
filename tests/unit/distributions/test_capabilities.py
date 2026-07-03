"""Tests for backend-neutral distribution capability metadata."""

from __future__ import annotations

import pytest
from bayeswire.distributions import Exponential, HalfNormal, Normal, StudentT, Truncated, Uniform
from bayeswire.distributions._capabilities import has_scalar_inverse_cdf
from bayeswire.distributions.core import Distribution, DistributionValue, LogProbability

from jaxstanv5._backends.jax.distributions import is_inverse_cdf, is_sampleable


class _InverseCdfWithoutLogProb:
    """Custom distribution with inverse-CDF methods but no log_prob support."""

    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def event_shape(self) -> tuple[int, ...]:
        return ()

    def sample(
        self,
        key: DistributionValue,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> DistributionValue:
        return 0.5

    def cdf(self, x: DistributionValue) -> DistributionValue:
        return x

    def icdf(self, p: DistributionValue) -> DistributionValue:
        return p


class _InverseCdfWithLogProb(_InverseCdfWithoutLogProb):
    """Custom distribution with full inverse-CDF and log_prob support."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return 0.0


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


def test_custom_distribution_without_log_prob_lacks_inverse_cdf_capability() -> None:
    distribution = _InverseCdfWithoutLogProb()

    assert has_scalar_inverse_cdf(distribution) is False
    assert is_inverse_cdf(distribution) is False
    assert is_sampleable(Truncated(distribution, lower=0.0)) is False


def test_custom_distribution_with_log_prob_has_inverse_cdf_capability() -> None:
    distribution = _InverseCdfWithLogProb()

    assert has_scalar_inverse_cdf(distribution) is True
    assert is_inverse_cdf(distribution) is True
    assert is_sampleable(Truncated(distribution, lower=0.0)) is True
