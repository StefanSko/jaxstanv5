"""Beta-binomial distribution."""

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
class BetaBinomial(DiscreteDistribution):
    """Beta-binomial distribution parameterized by count and beta concentrations."""

    total_count: DistributionParameter
    alpha: DistributionParameter
    beta: DistributionParameter

    def _total_count(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.total_count))

    def _alpha(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.alpha))

    def _beta(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.beta))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Beta-binomial parameters."""
        return jnp.broadcast_shapes(
            self._total_count().shape, self._alpha().shape, self._beta().shape
        )

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Beta-binomial draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Beta-binomial log-probability mass for ``x``."""
        raw_alpha = self._alpha()
        raw_beta = self._beta()
        dtype = jnp.result_type(raw_alpha, raw_beta, 1.0)
        total_count = jnp.asarray(_concrete_parameter(self.total_count), dtype=dtype)
        alpha = jnp.asarray(_concrete_parameter(self.alpha), dtype=dtype)
        beta = jnp.asarray(_concrete_parameter(self.beta), dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        integer_value = value == jnp.floor(value)
        integer_count = total_count == jnp.floor(total_count)
        support = (
            integer_value
            & integer_count
            & (value >= 0.0)
            & (total_count >= 0.0)
            & (value <= total_count)
            & (alpha > 0.0)
            & (beta > 0.0)
        )
        failures = total_count - value
        log_choose = gammaln(total_count + 1.0) - gammaln(value + 1.0) - gammaln(failures + 1.0)
        log_beta_observed = gammaln(value + alpha) + gammaln(failures + beta)
        log_beta_observed -= gammaln(total_count + alpha + beta)
        log_beta_prior = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        log_mass = log_choose + log_beta_observed - log_beta_prior
        return jnp.where(support, log_mass, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Beta-binomial samples with leading ``sample_shape`` dimensions."""
        probability_key, count_key = jax.random.split(key)
        shape = sample_shape + self.batch_shape()
        probability = jax.random.beta(count_key, a=self._alpha(), b=self._beta(), shape=shape)
        return jax.random.binomial(
            probability_key,
            n=self._total_count(),
            p=probability,
            shape=shape,
        )
