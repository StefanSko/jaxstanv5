"""Core distribution metadata protocols and type aliases."""

from typing import Protocol, cast, runtime_checkable


class SymbolicDistributionParameter:
    """Marker for declaration-time symbolic distribution fields."""


class DiscreteDistribution:
    """Marker for discrete distributions, which cannot be latent NUTS parameters."""


type DistributionValue = object
type DistributionParameter = object | SymbolicDistributionParameter
type LogProbability = object


def _concrete_parameter(value: DistributionParameter) -> object:
    """Treat a distribution field as concrete after symbolic field evaluation."""
    return cast(object, value)


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
        key: object,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> object:
        """Draw samples with leading ``sample_shape`` dimensions."""
        ...


@runtime_checkable
class InverseCdfDistribution(SampleableDistribution, Protocol):
    """Scalar distribution supporting inverse-CDF restricted sampling."""

    def cdf(self, x: DistributionValue) -> object:
        """Return element-wise cumulative probability at ``x``."""
        ...

    def icdf(self, p: DistributionValue) -> object:
        """Return element-wise inverse cumulative probability at ``p``."""
        ...
