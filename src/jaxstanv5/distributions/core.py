"""Core distribution metadata protocols and type aliases."""

from typing import Protocol, runtime_checkable


class SymbolicDistributionParameter:
    """Marker for declaration-time symbolic distribution fields."""


class DiscreteDistribution:
    """Marker for discrete distributions, which cannot be latent NUTS parameters."""


type DistributionValue = object
type DistributionParameter = object | SymbolicDistributionParameter
type LogProbability = object


def _concrete_parameter(value: DistributionParameter) -> DistributionValue:
    """Treat a distribution field as concrete after symbolic field evaluation."""
    return value


class Distribution(Protocol):
    """Distribution metadata consumed by numerical backends."""


@runtime_checkable
class SampleableDistribution(Distribution, Protocol):
    """Python compatibility protocol for distributions with sampling behavior.

    Must stay structurally aligned with the JAX backend dispatch protocol
    ``PythonSampleableDistribution`` so capability predicates match dispatch.
    """

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise log probability."""
        ...

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
    """Python compatibility protocol for scalar inverse-CDF behavior.

    Must stay structurally aligned with the JAX backend dispatch protocol
    ``PythonInverseCdfDistribution`` so capability predicates match dispatch.
    """

    def cdf(self, x: DistributionValue) -> DistributionValue:
        """Return element-wise cumulative probability at ``x``."""
        ...

    def icdf(self, p: DistributionValue) -> DistributionValue:
        """Return element-wise inverse cumulative probability at ``p``."""
        ...
