"""Exponential distribution."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstanv5.distributions.core import (
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class Exponential:
    """Exponential distribution parameterized by positive rate."""

    rate: DistributionParameter

    def _rate(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.rate))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Exponential parameters."""
        return self._rate().shape

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Exponential log-density for ``x``."""
        rate = self._rate()
        value = jnp.asarray(x)
        return jnp.where(value >= 0.0, jnp.log(rate) - rate * value, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Exponential samples with leading ``sample_shape`` dimensions."""
        rate = self._rate()
        standard = jax.random.exponential(key, shape=sample_shape + self.batch_shape())
        return standard / rate

    def cdf(self, x: DistributionValue) -> jax.Array:
        """Return element-wise Exponential cumulative probability at ``x``."""
        rate = self._rate()
        value = jnp.asarray(x)
        return jnp.where(value >= 0.0, -jnp.expm1(-rate * value), 0.0)

    def icdf(self, p: DistributionValue) -> jax.Array:
        """Return element-wise Exponential inverse cumulative probability at ``p``."""
        return -jnp.log1p(-jnp.asarray(p)) / self._rate()
