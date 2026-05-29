"""Core distribution types."""

from typing import Protocol, cast

import jax
from jax.typing import ArrayLike


class SymbolicDistributionParameter:
    """Marker for declaration-time symbolic distribution fields."""


type DistributionValue = ArrayLike
type DistributionParameter = ArrayLike | SymbolicDistributionParameter
type LogProbability = jax.Array


def _concrete_parameter(value: DistributionParameter) -> ArrayLike:
    """Treat a distribution field as concrete after symbolic field evaluation."""
    return cast(ArrayLike, value)


class Distribution(Protocol):
    """Probability distribution with an element-wise log-density."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise log-probability for ``x``."""
        ...


class SampleableDistribution(Distribution, Protocol):
    """Probability distribution that can draw prior-predictive samples."""

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw samples with leading ``sample_shape`` dimensions."""
        ...
