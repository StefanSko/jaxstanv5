"""Scalar continuous probability distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, ndtr, ndtri

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
        support = scale > 0.0
        safe_scale = jnp.where(support, scale, 1.0)
        standardized = (value - loc) / safe_scale
        log_density = -0.5 * standardized**2 - jnp.log(safe_scale) - 0.5 * math.log(2.0 * math.pi)
        return jnp.where(support, log_density, -jnp.inf)

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


@dataclass(frozen=True)
class HalfNormal:
    """Half-Normal distribution on non-negative real values."""

    scale: DistributionParameter

    def _scale(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.scale))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Half-Normal parameters."""
        return self._scale().shape

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Half-Normal draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Half-Normal log-density for ``x``."""
        scale = self._scale()
        value = jnp.asarray(x)
        valid_scale = scale > 0.0
        safe_scale = jnp.where(valid_scale, scale, 1.0)
        standardized = value / safe_scale
        log_density = 0.5 * math.log(2.0 / math.pi) - jnp.log(safe_scale) - 0.5 * standardized**2
        return jnp.where((value >= 0.0) & valid_scale, log_density, -jnp.inf)

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
        valid_params = (df > 0.0) & (scale > 0.0)
        safe_df = jnp.where(df > 0.0, df, 1.0)
        safe_scale = jnp.where(scale > 0.0, scale, 1.0)
        standardized = (value - loc) / safe_scale
        log_density = (
            gammaln(0.5 * (safe_df + 1.0))
            - gammaln(0.5 * safe_df)
            - 0.5 * jnp.log(safe_df * math.pi)
            - jnp.log(safe_scale)
            - 0.5 * (safe_df + 1.0) * jnp.log1p(standardized**2 / safe_df)
        )
        return jnp.where(valid_params, log_density, -jnp.inf)

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


@dataclass(frozen=True)
class Exponential:
    """Exponential distribution parameterized by positive rate."""

    rate: DistributionParameter

    def _rate(self) -> jax.Array:
        return jnp.asarray(_concrete_parameter(self.rate))

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for Exponential parameters."""
        return self._rate().shape

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Exponential draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise Exponential log-density for ``x``."""
        rate = self._rate()
        value = jnp.asarray(x)
        valid_rate = rate > 0.0
        safe_rate = jnp.where(valid_rate, rate, 1.0)
        log_density = jnp.log(safe_rate) - safe_rate * value
        return jnp.where((value >= 0.0) & valid_rate, log_density, -jnp.inf)

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

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise Uniform draws."""
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise uniform log-density for ``x``."""
        low, high = self._low_high()
        value = jnp.asarray(x)
        valid_bounds = high > low
        safe_width = jnp.where(valid_bounds, high - low, 1.0)
        in_support = (value >= low) & (value <= high) & valid_bounds
        return jnp.where(in_support, -jnp.log(safe_width), -jnp.inf)

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
