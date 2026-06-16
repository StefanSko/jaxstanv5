"""Core distribution metadata protocols and type aliases."""

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    import jax
    from jax.typing import ArrayLike


class SymbolicDistributionParameter:
    """Marker for declaration-time symbolic distribution fields."""


class DiscreteDistribution:
    """Marker for discrete distributions, which cannot be latent NUTS parameters."""


if TYPE_CHECKING:
    type DistributionValue = ArrayLike
    type DistributionParameter = ArrayLike | SymbolicDistributionParameter
    type LogProbability = jax.Array
else:
    type DistributionValue = object
    type DistributionParameter = object | SymbolicDistributionParameter
    type LogProbability = object


def _concrete_parameter(value: DistributionParameter) -> DistributionValue:
    """Treat a distribution field as concrete after symbolic field evaluation."""
    return cast(DistributionValue, value)


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
        key: DistributionValue,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> DistributionValue:
        """Draw samples with leading ``sample_shape`` dimensions."""
        ...


@runtime_checkable
class InverseCdfDistribution(SampleableDistribution, Protocol):
    """Scalar distribution supporting inverse-CDF restricted sampling."""

    def cdf(self, x: DistributionValue) -> DistributionValue:
        """Return element-wise cumulative probability at ``x``."""
        ...

    def icdf(self, p: DistributionValue) -> DistributionValue:
        """Return element-wise inverse cumulative probability at ``p``."""
        ...
