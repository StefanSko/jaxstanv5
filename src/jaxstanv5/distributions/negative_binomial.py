"""Negative-binomial distribution."""

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
class NegativeBinomial(DiscreteDistribution):
    """Negative-binomial NB2 distribution parameterized by mean and overdispersion."""

    mean: DistributionParameter
    overdispersion: DistributionParameter

    def _mean(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.mean))

    def _overdispersion(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.overdispersion))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Negative-binomial parameters."""
        return jnp.broadcast_shapes(self._mean().shape, self._overdispersion().shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Negative-binomial draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise NB2 log-probability mass for ``x``.

        The parameterization is ``E[x] = mean`` and
        ``Var[x] = mean + mean**2 / overdispersion``.
        """
        raw_mean = self._mean()
        raw_overdispersion = self._overdispersion()
        dtype = jnp.result_type(raw_mean, raw_overdispersion, 1.0)
        mean = jnp.asarray(_concrete_parameter(self.mean), dtype=dtype)
        overdispersion = jnp.asarray(_concrete_parameter(self.overdispersion), dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        integer_value = value == jnp.floor(value)
        support = integer_value & (value >= 0.0) & (mean > 0.0) & (overdispersion > 0.0)
        total = mean + overdispersion
        log_mass = gammaln(value + overdispersion) - gammaln(overdispersion) - gammaln(value + 1.0)
        log_mass = log_mass + xlogy(overdispersion, overdispersion / total)
        log_mass = log_mass + xlogy(value, mean / total)
        return jnp.where(support, log_mass, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Negative-binomial samples with leading ``sample_shape`` dimensions."""
        gamma_key, poisson_key = jax.random.split(key)
        shape = sample_shape + self.batch_shape()
        rate = jax.random.gamma(gamma_key, self._overdispersion(), shape=shape)
        rate = rate * self._mean() / self._overdispersion()
        return jax.random.poisson(poisson_key, lam=rate, shape=shape)
