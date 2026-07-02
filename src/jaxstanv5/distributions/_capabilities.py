"""Backend-neutral distribution capability predicates."""

from __future__ import annotations

from jaxstanv5.distributions.continuous import Exponential, HalfNormal, Normal, Uniform
from jaxstanv5.distributions.core import Distribution, InverseCdfDistribution
from jaxstanv5.distributions.truncated import Truncated


def has_scalar_inverse_cdf(distribution: Distribution) -> bool:
    """Return whether distribution metadata has scalar inverse-CDF support."""
    if isinstance(distribution, Truncated):
        return _base_has_scalar_inverse_cdf(distribution.base)
    return _base_has_scalar_inverse_cdf(distribution)


def _base_has_scalar_inverse_cdf(distribution: Distribution) -> bool:
    """Return whether non-Truncated distribution metadata has inverse-CDF support."""
    return isinstance(
        distribution,
        Normal | HalfNormal | Exponential | Uniform | InverseCdfDistribution,
    )
