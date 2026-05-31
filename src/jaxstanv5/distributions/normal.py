"""Normal distribution."""

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
class Normal:
    """Univariate normal distribution supporting JAX broadcasting."""

    loc: DistributionParameter
    scale: DistributionParameter

    def _loc_scale(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.loc)),
            jnp.asarray(_concrete_parameter(self.scale)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Normal parameters."""
        loc, scale = self._loc_scale()
        return jnp.broadcast_shapes(loc.shape, scale.shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Normal draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise normal log-density for ``x``."""
        loc, scale = self._loc_scale()
        value = jnp.asarray(x)
        standardized = (value - loc) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Normal samples with leading ``sample_shape`` dimensions."""
        loc, scale = self._loc_scale()
        standard = jax.random.normal(key, shape=sample_shape + self.batch_shape())
        return loc + scale * standard

    def cdf(self, x: DistributionValue) -> jax.Array:
        """Return element-wise Normal cumulative probability at ``x``."""
        loc, scale = self._loc_scale()
        return ndtr((jnp.asarray(x) - loc) / scale)

    def icdf(self, p: DistributionValue) -> jax.Array:
        """Return element-wise Normal inverse cumulative probability at ``p``."""
        loc, scale = self._loc_scale()
        return loc + scale * ndtri(jnp.asarray(p))
