"""Half-Normal distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from jaxstanv5.distributions.core import (
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class HalfNormal:
    """Half-Normal distribution on non-negative real values."""

    scale: DistributionParameter

    def _scale(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.scale))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Half-Normal parameters."""
        return self._scale().shape

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Half-Normal log-density for ``x``."""
        scale = self._scale()
        value = jnp.asarray(x)
        standardized = value / scale
        log_density = (
            0.5 * math.log(2.0 / math.pi) - jnp.log(scale) - 0.5 * standardized**2
        )
        return jnp.where(value >= 0.0, log_density, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Half-Normal samples with leading ``sample_shape`` dimensions."""
        scale = self._scale()
        standard = jax.random.normal(key, shape=sample_shape + self.batch_shape())
        return scale * jnp.abs(standard)

    def cdf(self, x: DistributionValue) -> jax.Array:
        """Return element-wise Half-Normal cumulative probability at ``x``."""
        value = jnp.asarray(x)
        probability = 2.0 * ndtr(value / self._scale()) - 1.0
        return jnp.where(value >= 0.0, probability, 0.0)

    def icdf(self, p: DistributionValue) -> jax.Array:
        """Return element-wise Half-Normal inverse cumulative probability at ``p``."""
        return self._scale() * ndtri(0.5 * (jnp.asarray(p) + 1.0))
