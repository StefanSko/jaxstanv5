"""Poisson distribution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions.core import (
    DiscreteDistribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class Poisson(DiscreteDistribution):
    """Poisson distribution parameterized by positive rate."""

    rate: DistributionParameter

    def _rate(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.rate))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Poisson parameters."""
        return self._rate().shape

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Poisson draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Poisson log-probability mass for ``x``."""
        rate = self._rate()
        value = jnp.asarray(x)
        integer_support = value == jnp.floor(value)
        support = (value >= 0.0) & integer_support & (rate > 0.0)
        log_mass = value * jnp.log(rate) - rate - gammaln(value + 1.0)
        return jnp.where(support, log_mass, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Poisson samples with leading ``sample_shape`` dimensions."""
        return jax.random.poisson(key, lam=self._rate(), shape=sample_shape + self.batch_shape())
