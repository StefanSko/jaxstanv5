"""Discrete count probability distributions."""

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
class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution parameterized by success probability."""

    probs: DistributionParameter

    def _probs(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.probs))

    def batch_shape(self) -> tuple[int, ...]:
        """Return non-sample dimensions for Bernoulli probabilities."""
        return self._probs().shape

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Bernoulli draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Bernoulli log-probability mass for ``x``."""
        probs = self._probs()
        dtype = jnp.result_type(probs, 1.0)
        probability = jnp.asarray(_concrete_parameter(self.probs), dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        integer_value = value == jnp.floor(value)
        support = integer_value & (value >= 0.0) & (value <= 1.0) & (probs >= 0.0) & (probs <= 1.0)
        log_mass = xlogy(value, probability) + xlogy(1.0 - value, 1.0 - probability)
        return jnp.where(support, log_mass, -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw Bernoulli samples with leading ``sample_shape`` dimensions."""
        return jax.random.bernoulli(
            key,
            p=self._probs(),
            shape=sample_shape + self.batch_shape(),
        ).astype(jnp.int32)


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
        probs = self._probs()
        dtype = jnp.result_type(probs, 1.0)
        total_count = jnp.asarray(_concrete_parameter(self.total_count), dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
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
