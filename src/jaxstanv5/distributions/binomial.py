"""Binomial distribution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, xlogy

from jaxstanv5.distributions.core import (
    DiscreteDistribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class Binomial(DiscreteDistribution):
    """Binomial distribution parameterized by total count and success probability."""

    total_count: DistributionParameter
    probs: DistributionParameter

    def _total_count(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.total_count))

    def _probs(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.probs))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Binomial parameters."""
        return jnp.broadcast_shapes(self._total_count().shape, self._probs().shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Binomial draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Binomial log-probability mass for ``x``."""
        total_count = self._total_count()
        probs = self._probs()
        value = jnp.asarray(x)
        integer_value = value == jnp.floor(value)
        integer_count = total_count == jnp.floor(total_count)
        support = (
            integer_value
            & integer_count
            & (value >= 0.0)
            & (total_count >= 0.0)
            & (value <= total_count)
            & (probs >= 0.0)
            & (probs <= 1.0)
        )
        failures = total_count - value
        log_mass = gammaln(total_count + 1.0) - gammaln(value + 1.0) - gammaln(failures + 1.0)
        log_mass = log_mass + xlogy(value, probs) + xlogy(failures, 1.0 - probs)
        return jnp.where(support, log_mass, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Binomial samples with leading ``sample_shape`` dimensions."""
        return jax.random.binomial(
            key,
            n=self._total_count(),
            p=self._probs(),
            shape=sample_shape + self.batch_shape(),
        )
