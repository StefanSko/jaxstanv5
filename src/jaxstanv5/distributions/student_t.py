"""Student-t distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from jaxstanv5.distributions.core import (
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class StudentT:
    """Student-t distribution parameterized by df, location, and scale."""

    df: DistributionParameter
    loc: DistributionParameter
    scale: DistributionParameter

    def _params(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.df)),
            jnp.asarray(_concrete_parameter(self.loc)),
            jnp.asarray(_concrete_parameter(self.scale)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Student-t parameters."""
        df, loc, scale = self._params()
        return jnp.broadcast_shapes(df.shape, loc.shape, scale.shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Student-t draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Student-t log-density for ``x``."""
        df, loc, scale = self._params()
        value = jnp.asarray(x)
        standardized = (value - loc) / scale
        return (
            gammaln(0.5 * (df + 1.0))
            - gammaln(0.5 * df)
            - 0.5 * jnp.log(df * math.pi)
            - jnp.log(scale)
            - 0.5 * (df + 1.0) * jnp.log1p(standardized**2 / df)
        )

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Student-t samples with leading ``sample_shape`` dimensions."""
        df, loc, scale = self._params()
        standard = jax.random.t(key, df, shape=sample_shape + self.batch_shape())
        return loc + scale * standard
