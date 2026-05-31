"""Beta distribution."""

from __future__ import annotations

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
class Beta:
    """Beta distribution parameterized by positive concentrations."""

    alpha: DistributionParameter
    beta: DistributionParameter

    def _alpha_beta(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.alpha)),
            jnp.asarray(_concrete_parameter(self.beta)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Beta parameters."""
        alpha, beta = self._alpha_beta()
        return jnp.broadcast_shapes(alpha.shape, beta.shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Beta draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Beta log-density for ``x``."""
        raw_alpha, raw_beta = self._alpha_beta()
        dtype = jnp.result_type(raw_alpha, raw_beta, x, 1.0)
        alpha = jnp.asarray(_concrete_parameter(self.alpha), dtype=dtype)
        beta = jnp.asarray(_concrete_parameter(self.beta), dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        support = (value > 0.0) & (value < 1.0) & (alpha > 0.0) & (beta > 0.0)
        safe_value = jnp.clip(value, jnp.finfo(dtype).tiny, 1.0 - jnp.finfo(dtype).eps)
        log_normalizer = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        log_density = (alpha - 1.0) * jnp.log(safe_value)
        log_density += (beta - 1.0) * jnp.log1p(-safe_value)
        log_density -= log_normalizer
        return jnp.where(support, log_density, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Beta samples with leading ``sample_shape`` dimensions."""
        alpha, beta = self._alpha_beta()
        return jax.random.beta(key, a=alpha, b=beta, shape=sample_shape + self.batch_shape())
