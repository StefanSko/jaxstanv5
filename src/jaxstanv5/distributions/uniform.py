"""Uniform distribution."""

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
class Uniform:
    """Continuous uniform distribution supporting JAX broadcasting."""

    low: DistributionParameter
    high: DistributionParameter

    def _low_high(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.low)),
            jnp.asarray(_concrete_parameter(self.high)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Uniform parameters."""
        low, high = self._low_high()
        return jnp.broadcast_shapes(low.shape, high.shape)

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise uniform log-density for ``x``."""
        low, high = self._low_high()
        value = jnp.asarray(x)
        in_support = (value >= low) & (value <= high)
        return jnp.where(in_support, -jnp.log(high - low), -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Uniform samples with leading ``sample_shape`` dimensions."""
        low, high = self._low_high()
        standard = jax.random.uniform(key, shape=sample_shape + self.batch_shape())
        return low + (high - low) * standard

    def cdf(self, x: DistributionValue) -> jax.Array:
        """Return element-wise Uniform cumulative probability at ``x``."""
        low, high = self._low_high()
        standardized = (jnp.asarray(x) - low) / (high - low)
        return jnp.clip(standardized, 0.0, 1.0)

    def icdf(self, p: DistributionValue) -> jax.Array:
        """Return element-wise Uniform inverse cumulative probability at ``p``."""
        low, high = self._low_high()
        return low + (high - low) * jnp.asarray(p)
