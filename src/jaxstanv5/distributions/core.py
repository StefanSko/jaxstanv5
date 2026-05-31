"""Core distribution types."""

from typing import Protocol, cast, runtime_checkable

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


@runtime_checkable
class SampleableDistribution(Distribution, Protocol):
    """Probability distribution that can draw prior-predictive samples."""

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample, non-event dimensions."""
        ...

    def event_shape(self) -> tuple[int, ...]:
        """Return per-draw event dimensions."""
        ...

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw samples with leading ``sample_shape`` dimensions."""
        ...


@runtime_checkable
class InverseCdfDistribution(SampleableDistribution, Protocol):
    """Scalar distribution supporting inverse-CDF restricted sampling."""

    def cdf(self, x: DistributionValue) -> jax.Array:
        """Return element-wise cumulative probability at ``x``."""
        ...

    def icdf(self, p: DistributionValue) -> jax.Array:
        """Return element-wise inverse cumulative probability at ``p``."""
        ...
