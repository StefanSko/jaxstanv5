"""JAX implementations of distribution operations."""

from __future__ import annotations

import math
from typing import Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import gammaln, log_ndtr, ndtr, ndtri, xlogy

from jaxstanv5.distributions.continuous import (
    Beta,
    Exponential,
    HalfNormal,
    Normal,
    StudentT,
    Uniform,
)
from jaxstanv5.distributions.core import Distribution, DistributionParameter, DistributionValue
from jaxstanv5.distributions.counts import (
    Bernoulli,
    BetaBinomial,
    Binomial,
    NegativeBinomial,
    Poisson,
)
from jaxstanv5.distributions.multivariate import MultivariateNormal
from jaxstanv5.distributions.ordinal import OrderedLogistic
from jaxstanv5.distributions.truncated import Truncated


@runtime_checkable
class PythonLogProbDistribution(Protocol):
    """Compatibility protocol for Python-defined JAX log-probability providers."""

    def log_prob(self, x: DistributionValue) -> object:
        """Return element-wise log probability."""
        ...


@runtime_checkable
class PythonSampleableDistribution(PythonLogProbDistribution, Protocol):
    """Compatibility protocol for Python-defined JAX sample providers."""

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample, non-event dimensions."""
        ...

    def event_shape(self) -> tuple[int, ...]:
        """Return per-draw event dimensions."""
        ...

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> object:
        """Draw samples with leading sample dimensions."""
        ...


@runtime_checkable
class PythonInverseCdfDistribution(PythonSampleableDistribution, Protocol):
    """Compatibility protocol for Python-defined JAX inverse-CDF providers."""

    def cdf(self, x: DistributionValue) -> object:
        """Return cumulative probability."""
        ...

    def icdf(self, p: DistributionValue) -> object:
        """Return inverse cumulative probability."""
        ...


def _asarray(value: DistributionParameter) -> jax.Array:
    return jnp.asarray(value)


def _normal_loc_scale(distribution: Normal) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.loc), _asarray(distribution.scale)


def _halfnormal_scale(distribution: HalfNormal) -> jax.Array:
    return _asarray(distribution.scale)


def _student_t_params(distribution: StudentT) -> tuple[jax.Array, jax.Array, jax.Array]:
    return _asarray(distribution.df), _asarray(distribution.loc), _asarray(distribution.scale)


def _exponential_rate(distribution: Exponential) -> jax.Array:
    return _asarray(distribution.rate)


def _uniform_low_high(distribution: Uniform) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.low), _asarray(distribution.high)


def _beta_alpha_beta(distribution: Beta) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.alpha), _asarray(distribution.beta)


def _truncated_bounds(distribution: Truncated) -> tuple[jax.Array | None, jax.Array | None]:
    lower = None if distribution.lower is None else _asarray(distribution.lower)
    upper = None if distribution.upper is None else _asarray(distribution.upper)
    return lower, upper


def _bernoulli_probs(distribution: Bernoulli) -> jax.Array:
    return _asarray(distribution.probs)


def _poisson_rate(distribution: Poisson) -> jax.Array:
    return _asarray(distribution.rate)


def _binomial_total_count_probs(distribution: Binomial) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.total_count), _asarray(distribution.probs)


def _beta_binomial_params(distribution: BetaBinomial) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        _asarray(distribution.total_count),
        _asarray(distribution.alpha),
        _asarray(distribution.beta),
    )


def _negative_binomial_params(distribution: NegativeBinomial) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.mean), _asarray(distribution.overdispersion)


def _mvn_mean_tril(distribution: MultivariateNormal) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.mean), _asarray(distribution.scale_tril)


def _ordered_logistic_eta_cutpoints(
    distribution: OrderedLogistic,
) -> tuple[jax.Array, jax.Array]:
    return _asarray(distribution.eta), _asarray(distribution.cutpoints)


def batch_shape(distribution: Distribution) -> tuple[int, ...]:
    """Return non-sample, non-event dimensions for a distribution."""
    if isinstance(distribution, Normal):
        loc, scale = _normal_loc_scale(distribution)
        return jnp.broadcast_shapes(loc.shape, scale.shape)
    if isinstance(distribution, HalfNormal):
        return _halfnormal_scale(distribution).shape
    if isinstance(distribution, StudentT):
        df, loc, scale = _student_t_params(distribution)
        return jnp.broadcast_shapes(df.shape, loc.shape, scale.shape)
    if isinstance(distribution, Exponential):
        return _exponential_rate(distribution).shape
    if isinstance(distribution, Uniform):
        low, high = _uniform_low_high(distribution)
        return jnp.broadcast_shapes(low.shape, high.shape)
    if isinstance(distribution, Beta):
        alpha, beta = _beta_alpha_beta(distribution)
        return jnp.broadcast_shapes(alpha.shape, beta.shape)
    if isinstance(distribution, Truncated):
        lower, upper = _truncated_bounds(distribution)
        shapes = [batch_shape(distribution.base)]
        if lower is not None:
            shapes.append(lower.shape)
        if upper is not None:
            shapes.append(upper.shape)
        return jnp.broadcast_shapes(*shapes)
    if isinstance(distribution, Bernoulli):
        return _bernoulli_probs(distribution).shape
    if isinstance(distribution, Poisson):
        return _poisson_rate(distribution).shape
    if isinstance(distribution, Binomial):
        total_count, probs = _binomial_total_count_probs(distribution)
        return jnp.broadcast_shapes(total_count.shape, probs.shape)
    if isinstance(distribution, BetaBinomial):
        total_count, alpha, beta = _beta_binomial_params(distribution)
        return jnp.broadcast_shapes(total_count.shape, alpha.shape, beta.shape)
    if isinstance(distribution, NegativeBinomial):
        mean, overdispersion = _negative_binomial_params(distribution)
        return jnp.broadcast_shapes(mean.shape, overdispersion.shape)
    if isinstance(distribution, MultivariateNormal):
        mean, scale_tril = _mvn_mean_tril(distribution)
        mean_batch_shape = () if mean.ndim == 0 else mean.shape[:-1]
        scale_batch_shape = scale_tril.shape[:-2]
        return jnp.broadcast_shapes(mean_batch_shape, scale_batch_shape)
    if isinstance(distribution, OrderedLogistic):
        eta, cutpoints = _ordered_logistic_eta_cutpoints(distribution)
        if cutpoints.ndim == 0:
            raise ValueError("OrderedLogistic cutpoints must be a vector")
        return jnp.broadcast_shapes(eta.shape, cutpoints.shape[:-1])
    if isinstance(distribution, PythonSampleableDistribution):
        return distribution.batch_shape()
    raise TypeError(f"Distribution has no JAX batch_shape support: {type(distribution).__name__}")


def event_shape(distribution: Distribution) -> tuple[int, ...]:
    """Return per-draw event dimensions for a distribution."""
    if isinstance(
        distribution,
        Normal
        | HalfNormal
        | StudentT
        | Exponential
        | Uniform
        | Beta
        | Bernoulli
        | Poisson
        | Binomial
        | BetaBinomial
        | NegativeBinomial
        | OrderedLogistic,
    ):
        return ()
    if isinstance(distribution, Truncated):
        return event_shape(distribution.base)
    if isinstance(distribution, MultivariateNormal):
        _, scale_tril = _mvn_mean_tril(distribution)
        return (scale_tril.shape[-1],)
    if isinstance(distribution, PythonSampleableDistribution):
        return distribution.event_shape()
    raise TypeError(f"Distribution has no JAX event_shape support: {type(distribution).__name__}")


def is_sampleable(distribution: Distribution) -> bool:
    """Return whether the JAX backend can sample from ``distribution``."""
    if isinstance(distribution, Truncated):
        return is_inverse_cdf(distribution.base)
    return isinstance(
        distribution,
        Normal
        | HalfNormal
        | StudentT
        | Exponential
        | Uniform
        | Beta
        | Bernoulli
        | Poisson
        | Binomial
        | BetaBinomial
        | NegativeBinomial
        | MultivariateNormal
        | OrderedLogistic
        | PythonSampleableDistribution,
    )


def is_inverse_cdf(distribution: Distribution) -> bool:
    """Return whether the JAX backend has scalar inverse-CDF support."""
    if isinstance(distribution, Truncated):
        return is_inverse_cdf(distribution.base)
    return isinstance(
        distribution,
        Normal | HalfNormal | Exponential | Uniform | PythonInverseCdfDistribution,
    )


def validate_scale_tril(scale_tril: jax.Array, *, name: str = "scale_tril") -> None:
    """Validate a concrete lower-triangular Cholesky scale matrix."""
    if scale_tril.ndim < 2:
        raise ValueError(f"{name} must be a matrix or batched matrix")
    if scale_tril.shape[-1] != scale_tril.shape[-2]:
        raise ValueError(f"{name} must be square")
    diagonal = jnp.diagonal(scale_tril, axis1=-2, axis2=-1)
    try:
        lower_triangular = bool(jnp.allclose(scale_tril, jnp.tril(scale_tril)))
        positive_diagonal = bool(jnp.all(diagonal > 0.0))
    except jax.errors.TracerBoolConversionError:
        return
    if not lower_triangular:
        raise ValueError(
            f"{name} must be lower-triangular; use jnp.linalg.cholesky(cov) "
            "when starting from a covariance matrix"
        )
    if not positive_diagonal:
        raise ValueError(f"{name} diagonal entries must be strictly positive")


def _mvn_event_value(value: jax.Array, event_size: int) -> jax.Array:
    if value.ndim == 0:
        if event_size == 1:
            return jnp.reshape(value, (1,))
        raise ValueError("MultivariateNormal values must have a trailing event dimension")
    if value.shape[-1] != event_size:
        raise ValueError(
            "MultivariateNormal values must have trailing dimension "
            f"{event_size}, got {value.shape[-1]}"
        )
    return value


def _ordered_logistic_category_probabilities(distribution: OrderedLogistic) -> jax.Array:
    eta, cutpoints = _ordered_logistic_eta_cutpoints(distribution)
    if cutpoints.ndim == 0:
        raise ValueError("OrderedLogistic cutpoints must be a vector")
    if cutpoints.shape[-1] < 1:
        raise ValueError("OrderedLogistic requires at least one cutpoint")
    shape = batch_shape(distribution)
    eta_broadcast = jnp.broadcast_to(eta, shape)
    cutpoints_broadcast = jnp.broadcast_to(cutpoints, shape + cutpoints.shape[-1:])
    cumulative = jax.nn.sigmoid(cutpoints_broadcast - eta_broadcast[..., None])
    first = cumulative[..., :1]
    middle = cumulative[..., 1:] - cumulative[..., :-1]
    last = 1.0 - cumulative[..., -1:]
    return jnp.concatenate((first, middle, last), axis=-1)


def _validate_scalar_event_truncated(distribution: Truncated) -> None:
    if event_shape(distribution.base) != ():
        raise TypeError("Truncated distributions require scalar-event base distributions")


def _truncated_probability_bounds(distribution: Truncated) -> tuple[jax.Array, jax.Array]:
    lower, upper = _truncated_bounds(distribution)
    lower_probability = jnp.asarray(0.0) if lower is None else cdf(distribution.base, lower)
    upper_probability = jnp.asarray(1.0) if upper is None else cdf(distribution.base, upper)
    return lower_probability, upper_probability


def _truncated_value_in_bounds(distribution: Truncated, value: jax.Array) -> jax.Array:
    lower, upper = _truncated_bounds(distribution)
    above_lower = jnp.ones_like(value, dtype=bool) if lower is None else value >= lower
    below_upper = jnp.ones_like(value, dtype=bool) if upper is None else value <= upper
    return above_lower & below_upper


def _log_sub_exp(log_left: jax.Array, log_right: jax.Array) -> jax.Array:
    """Return log(exp(log_left) - exp(log_right)) for log_left >= log_right."""
    ratio = jnp.exp(log_right - log_left)
    safe_ratio = jnp.minimum(ratio, 1.0)
    return log_left + jnp.log1p(-safe_ratio)


def _normal_log_interval_probability(
    distribution: Normal,
    lower: jax.Array | None,
    upper: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Return stable log probability that a Normal lies in the truncation interval."""
    loc, scale = _normal_loc_scale(distribution)
    valid_scale = scale > 0.0
    safe_scale = jnp.where(valid_scale, scale, 1.0)

    if lower is None:
        if upper is None:
            return jnp.zeros_like(loc + safe_scale), valid_scale
        z_upper = (upper - loc) / safe_scale
        return log_ndtr(z_upper), valid_scale
    z_lower = (lower - loc) / safe_scale
    if upper is None:
        return log_ndtr(-z_lower), valid_scale

    z_upper = (upper - loc) / safe_scale
    valid_bounds = (upper > lower) & valid_scale
    use_survival = z_lower > 0.0
    cdf_lower = jnp.where(use_survival, -1.0, z_lower)
    cdf_upper = jnp.where(use_survival, 0.0, z_upper)
    survival_lower = jnp.where(use_survival, z_lower, 0.0)
    survival_upper = jnp.where(use_survival, z_upper, 1.0)
    log_cdf_difference = _log_sub_exp(log_ndtr(cdf_upper), log_ndtr(cdf_lower))
    log_survival_difference = _log_sub_exp(log_ndtr(-survival_lower), log_ndtr(-survival_upper))
    log_probability = jnp.where(use_survival, log_survival_difference, log_cdf_difference)
    return log_probability, valid_bounds


def _truncated_log_normalizer(distribution: Truncated) -> tuple[jax.Array, jax.Array]:
    lower, upper = _truncated_bounds(distribution)
    if isinstance(distribution.base, Normal):
        return _normal_log_interval_probability(distribution.base, lower, upper)

    lower_probability, upper_probability = _truncated_probability_bounds(distribution)
    normalizer = upper_probability - lower_probability
    valid_normalizer = normalizer > 0.0
    safe_normalizer = jnp.where(valid_normalizer, normalizer, 1.0)
    return jnp.log(safe_normalizer), valid_normalizer


def _normal_truncated_cdf(distribution: Truncated, x: DistributionValue) -> jax.Array:
    base = cast(Normal, distribution.base)
    lower, upper = _truncated_bounds(distribution)
    loc, scale = _normal_loc_scale(base)
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    value = jnp.asarray(x)
    if lower is not None:
        value = jnp.maximum(value, lower)
    if upper is not None:
        value = jnp.minimum(value, upper)
    z_value = (value - loc) / safe_scale

    if lower is None:
        if upper is None:
            return jnp.ones_like(value)
        z_upper = (upper - loc) / safe_scale
        probability = jnp.exp(log_ndtr(z_value) - log_ndtr(z_upper))
        return jnp.clip(probability, 0.0, 1.0)

    z_lower = (lower - loc) / safe_scale
    if upper is None:
        log_survival_lower = log_ndtr(-z_lower)
        log_survival_value = log_ndtr(-z_value)
        probability = jnp.exp(
            _log_sub_exp(log_survival_lower, log_survival_value) - log_survival_lower
        )
        return jnp.clip(probability, 0.0, 1.0)

    z_upper = (upper - loc) / safe_scale
    use_survival = z_lower > 0.0
    cdf_lower = jnp.where(use_survival, -1.0, z_lower)
    cdf_value = jnp.where(use_survival, -0.5, z_value)
    cdf_upper = jnp.where(use_survival, 0.0, z_upper)
    survival_lower = jnp.where(use_survival, z_lower, 0.0)
    survival_value = jnp.where(use_survival, z_value, 0.5)
    survival_upper = jnp.where(use_survival, z_upper, 1.0)
    log_cdf_numerator = _log_sub_exp(log_ndtr(cdf_value), log_ndtr(cdf_lower))
    log_cdf_denominator = _log_sub_exp(log_ndtr(cdf_upper), log_ndtr(cdf_lower))
    log_survival_numerator = _log_sub_exp(log_ndtr(-survival_lower), log_ndtr(-survival_value))
    log_survival_denominator = _log_sub_exp(log_ndtr(-survival_lower), log_ndtr(-survival_upper))
    cdf_probability = jnp.exp(log_cdf_numerator - log_cdf_denominator)
    survival_probability = jnp.exp(log_survival_numerator - log_survival_denominator)
    probability = jnp.where(use_survival, survival_probability, cdf_probability)
    return jnp.clip(probability, 0.0, 1.0)


def _normal_truncated_icdf(distribution: Truncated, p: DistributionValue) -> jax.Array:
    base = cast(Normal, distribution.base)
    lower, upper = _truncated_bounds(distribution)
    loc, scale = _normal_loc_scale(base)
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    probability = jnp.clip(jnp.asarray(p), 0.0, 1.0)

    if lower is None:
        if upper is None:
            return loc + safe_scale * ndtri(probability)
        z_upper = (upper - loc) / safe_scale
        log_base_probability = log_ndtr(z_upper) + jnp.log(probability)
        return loc + safe_scale * ndtri(jnp.exp(log_base_probability))

    z_lower = (lower - loc) / safe_scale
    if upper is None:
        log_survival = log_ndtr(-z_lower) + jnp.log1p(-probability)
        return loc + safe_scale * -ndtri(jnp.exp(log_survival))

    z_upper = (upper - loc) / safe_scale
    use_survival = z_lower > 0.0
    cdf_lower = jnp.where(use_survival, -1.0, z_lower)
    cdf_upper = jnp.where(use_survival, 0.0, z_upper)
    log_cdf_lower = log_ndtr(cdf_lower)
    log_cdf_upper = log_ndtr(cdf_upper)
    cdf_ratio = jnp.exp(log_cdf_lower - log_cdf_upper)
    log_cdf_value = log_cdf_upper + jnp.log(cdf_ratio + probability * (1.0 - cdf_ratio))
    cdf_value = loc + safe_scale * ndtri(jnp.exp(log_cdf_value))

    survival_lower = jnp.where(use_survival, z_lower, 0.0)
    survival_upper = jnp.where(use_survival, z_upper, 1.0)
    log_survival_lower = log_ndtr(-survival_lower)
    log_survival_upper = log_ndtr(-survival_upper)
    survival_ratio = jnp.exp(log_survival_upper - log_survival_lower)
    log_survival_value = log_survival_lower + jnp.log1p(-probability * (1.0 - survival_ratio))
    survival_value = loc + safe_scale * -ndtri(jnp.exp(log_survival_value))
    return jnp.where(use_survival, survival_value, cdf_value)


def log_prob(distribution: Distribution, x: DistributionValue) -> jax.Array:
    """Return element-wise/event-wise log probability with the JAX backend."""
    if isinstance(distribution, Normal):
        loc, scale = _normal_loc_scale(distribution)
        value = jnp.asarray(x)
        support = scale > 0.0
        safe_scale = jnp.where(support, scale, 1.0)
        standardized = (value - loc) / safe_scale
        log_density = -0.5 * standardized**2 - jnp.log(safe_scale) - 0.5 * math.log(2.0 * math.pi)
        return jnp.where(support, log_density, -jnp.inf)
    if isinstance(distribution, HalfNormal):
        scale = _halfnormal_scale(distribution)
        value = jnp.asarray(x)
        valid_scale = scale > 0.0
        safe_scale = jnp.where(valid_scale, scale, 1.0)
        standardized = value / safe_scale
        log_density = 0.5 * math.log(2.0 / math.pi) - jnp.log(safe_scale) - 0.5 * standardized**2
        return jnp.where((value >= 0.0) & valid_scale, log_density, -jnp.inf)
    if isinstance(distribution, StudentT):
        df, loc, scale = _student_t_params(distribution)
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
    if isinstance(distribution, Exponential):
        rate = _exponential_rate(distribution)
        value = jnp.asarray(x)
        valid_rate = rate > 0.0
        safe_rate = jnp.where(valid_rate, rate, 1.0)
        log_density = jnp.log(safe_rate) - safe_rate * value
        return jnp.where((value >= 0.0) & valid_rate, log_density, -jnp.inf)
    if isinstance(distribution, Uniform):
        low, high = _uniform_low_high(distribution)
        value = jnp.asarray(x)
        valid_bounds = high > low
        safe_width = jnp.where(valid_bounds, high - low, 1.0)
        in_support = (value >= low) & (value <= high) & valid_bounds
        return jnp.where(in_support, -jnp.log(safe_width), -jnp.inf)
    if isinstance(distribution, Beta):
        raw_alpha, raw_beta = _beta_alpha_beta(distribution)
        dtype = jnp.result_type(raw_alpha, raw_beta, x, 1.0)
        alpha = jnp.asarray(distribution.alpha, dtype=dtype)
        beta = jnp.asarray(distribution.beta, dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        support = (value > 0.0) & (value < 1.0) & (alpha > 0.0) & (beta > 0.0)
        safe_value = jnp.clip(value, jnp.finfo(dtype).tiny, 1.0 - jnp.finfo(dtype).eps)
        log_normalizer = gammaln(alpha) + gammaln(beta) - gammaln(alpha + beta)
        log_density = (alpha - 1.0) * jnp.log(safe_value)
        log_density += (beta - 1.0) * jnp.log1p(-safe_value)
        log_density -= log_normalizer
        return jnp.where(support, log_density, -jnp.inf)
    if isinstance(distribution, Truncated):
        _validate_scalar_event_truncated(distribution)
        value = jnp.asarray(x)
        log_normalizer, valid_normalizer = _truncated_log_normalizer(distribution)
        safe_log_normalizer = jnp.where(valid_normalizer, log_normalizer, 0.0)
        truncated_log_density = log_prob(distribution.base, value) - safe_log_normalizer
        support = _truncated_value_in_bounds(distribution, value) & valid_normalizer
        return jnp.where(support, truncated_log_density, -jnp.inf)
    if isinstance(distribution, Bernoulli):
        probs = _bernoulli_probs(distribution)
        dtype = jnp.result_type(probs, 1.0)
        probability = jnp.asarray(distribution.probs, dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        integer_value = value == jnp.floor(value)
        support = integer_value & (value >= 0.0) & (value <= 1.0) & (probs >= 0.0) & (probs <= 1.0)
        log_mass = xlogy(value, probability) + xlogy(1.0 - value, 1.0 - probability)
        return jnp.where(support, log_mass, -jnp.inf)
    if isinstance(distribution, Poisson):
        rate = _poisson_rate(distribution)
        dtype = jnp.result_type(rate, 1.0)
        value = jnp.asarray(x, dtype=dtype)
        integer_support = value == jnp.floor(value)
        valid_rate = rate > 0.0
        support = (value >= 0.0) & integer_support & valid_rate
        safe_rate = jnp.where(valid_rate, rate, 1.0)
        log_mass = xlogy(value, safe_rate) - safe_rate - gammaln(value + 1.0)
        return jnp.where(support, log_mass, -jnp.inf)
    if isinstance(distribution, Binomial):
        total_count_raw, probs = _binomial_total_count_probs(distribution)
        dtype = jnp.result_type(probs, 1.0)
        total_count = jnp.asarray(total_count_raw, dtype=dtype)
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
    if isinstance(distribution, BetaBinomial):
        raw_count, raw_alpha, raw_beta = _beta_binomial_params(distribution)
        dtype = jnp.result_type(raw_alpha, raw_beta, 1.0)
        total_count = jnp.asarray(raw_count, dtype=dtype)
        alpha = jnp.asarray(raw_alpha, dtype=dtype)
        beta = jnp.asarray(raw_beta, dtype=dtype)
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
    if isinstance(distribution, NegativeBinomial):
        raw_mean, raw_overdispersion = _negative_binomial_params(distribution)
        dtype = jnp.result_type(raw_mean, raw_overdispersion, 1.0)
        mean = jnp.asarray(raw_mean, dtype=dtype)
        overdispersion = jnp.asarray(raw_overdispersion, dtype=dtype)
        value = jnp.asarray(x, dtype=dtype)
        integer_value = value == jnp.floor(value)
        support = integer_value & (value >= 0.0) & (mean > 0.0) & (overdispersion > 0.0)
        total = mean + overdispersion
        log_mass = gammaln(value + overdispersion) - gammaln(overdispersion) - gammaln(value + 1.0)
        log_mass = log_mass + xlogy(overdispersion, overdispersion / total)
        log_mass = log_mass + xlogy(value, mean / total)
        return jnp.where(support, log_mass, -jnp.inf)
    if isinstance(distribution, MultivariateNormal):
        mean, scale_tril = _mvn_mean_tril(distribution)
        size = scale_tril.shape[-1]
        value = _mvn_event_value(jnp.asarray(x), size)
        scale_tril = jnp.tril(scale_tril)
        delta = value - mean
        shape = jnp.broadcast_shapes(delta.shape[:-1], scale_tril.shape[:-2])
        delta = jnp.broadcast_to(delta, shape + (size,))
        scale_tril = jnp.broadcast_to(scale_tril, shape + (size, size))
        diagonal = jnp.diagonal(scale_tril, axis1=-2, axis2=-1)
        valid_diagonal = jnp.all(diagonal > 0.0, axis=-1)
        safe_diagonal = jnp.where(diagonal > 0.0, diagonal, 1.0)
        diagonal_index = jnp.arange(size)
        safe_scale_tril = scale_tril.at[..., diagonal_index, diagonal_index].set(safe_diagonal)
        flat_delta = delta.reshape((-1, size))
        flat_scale_tril = safe_scale_tril.reshape((-1, size, size))
        solved = jax.vmap(lambda tril, row: solve_triangular(tril, row, lower=True))(
            flat_scale_tril,
            flat_delta,
        )
        solved = solved.reshape(shape + (size,))
        quadratic = jnp.sum(solved**2, axis=-1)
        log_det = jnp.sum(jnp.log(safe_diagonal), axis=-1)
        log_probability = -0.5 * quadratic - log_det - 0.5 * size * math.log(2.0 * math.pi)
        return jnp.where(valid_diagonal, jnp.reshape(log_probability, shape), -jnp.inf)
    if isinstance(distribution, OrderedLogistic):
        eta, cutpoints = _ordered_logistic_eta_cutpoints(distribution)
        probabilities = _ordered_logistic_category_probabilities(distribution)
        category_count = cutpoints.shape[-1] + 1
        shape = jnp.broadcast_shapes(probabilities.shape[:-1], jnp.asarray(x).shape)
        probabilities_broadcast = jnp.broadcast_to(probabilities, shape + (category_count,))
        value = jnp.broadcast_to(jnp.asarray(x), shape)
        integer_value = value == jnp.floor(value)
        in_range = (value >= 0.0) & (value <= float(category_count - 1))
        safe_index = jnp.clip(value, 0.0, float(category_count - 1)).astype(jnp.int32)
        selected_probability = jnp.take_along_axis(
            probabilities_broadcast,
            safe_index[..., None],
            axis=-1,
        )[..., 0]
        ordered_cutpoints = jnp.all(cutpoints[..., 1:] > cutpoints[..., :-1], axis=-1)
        ordered_broadcast = jnp.broadcast_to(ordered_cutpoints, probabilities.shape[:-1])
        ordered_broadcast = jnp.broadcast_to(ordered_broadcast, shape)
        support = integer_value & in_range & ordered_broadcast & (selected_probability > 0.0)
        dtype = jnp.result_type(eta, cutpoints, 1.0)
        safe_probability = jnp.clip(selected_probability, jnp.finfo(dtype).tiny, 1.0)
        return jnp.where(support, jnp.log(safe_probability), -jnp.inf)
    if isinstance(distribution, PythonLogProbDistribution):
        return cast(jax.Array, distribution.log_prob(x))
    raise TypeError(f"Distribution has no JAX log_prob support: {type(distribution).__name__}")


def sample(
    distribution: Distribution,
    key: jax.Array,
    *,
    sample_shape: tuple[int, ...] = (),
) -> jax.Array:
    """Draw samples with the JAX backend."""
    if isinstance(distribution, Normal):
        loc, scale = _normal_loc_scale(distribution)
        standard = jax.random.normal(key, shape=sample_shape + batch_shape(distribution))
        return loc + scale * standard
    if isinstance(distribution, HalfNormal):
        scale = _halfnormal_scale(distribution)
        standard = jax.random.normal(key, shape=sample_shape + batch_shape(distribution))
        return scale * jnp.abs(standard)
    if isinstance(distribution, StudentT):
        df, loc, scale = _student_t_params(distribution)
        standard = jax.random.t(key, df, shape=sample_shape + batch_shape(distribution))
        return loc + scale * standard
    if isinstance(distribution, Exponential):
        rate = _exponential_rate(distribution)
        standard = jax.random.exponential(key, shape=sample_shape + batch_shape(distribution))
        return standard / rate
    if isinstance(distribution, Uniform):
        low, high = _uniform_low_high(distribution)
        standard = jax.random.uniform(key, shape=sample_shape + batch_shape(distribution))
        return low + (high - low) * standard
    if isinstance(distribution, Beta):
        alpha, beta = _beta_alpha_beta(distribution)
        return jax.random.beta(key, a=alpha, b=beta, shape=sample_shape + batch_shape(distribution))
    if isinstance(distribution, Truncated):
        _validate_scalar_event_truncated(distribution)
        probabilities = jax.random.uniform(key, shape=sample_shape + batch_shape(distribution))
        return icdf(distribution, probabilities)
    if isinstance(distribution, Bernoulli):
        return jax.random.bernoulli(
            key,
            p=_bernoulli_probs(distribution),
            shape=sample_shape + batch_shape(distribution),
        ).astype(jnp.int32)
    if isinstance(distribution, Poisson):
        return jax.random.poisson(
            key,
            lam=_poisson_rate(distribution),
            shape=sample_shape + batch_shape(distribution),
        )
    if isinstance(distribution, Binomial):
        total_count, probs = _binomial_total_count_probs(distribution)
        return jax.random.binomial(
            key,
            n=total_count,
            p=probs,
            shape=sample_shape + batch_shape(distribution),
        ).astype(jnp.int32)
    if isinstance(distribution, BetaBinomial):
        total_count, alpha, beta = _beta_binomial_params(distribution)
        probability_key, count_key = jax.random.split(key)
        shape = sample_shape + batch_shape(distribution)
        probability = jax.random.beta(probability_key, a=alpha, b=beta, shape=shape)
        return jax.random.binomial(count_key, n=total_count, p=probability, shape=shape).astype(
            jnp.int32
        )
    if isinstance(distribution, NegativeBinomial):
        mean, overdispersion = _negative_binomial_params(distribution)
        gamma_key, poisson_key = jax.random.split(key)
        shape = sample_shape + batch_shape(distribution)
        rate = jax.random.gamma(gamma_key, overdispersion, shape=shape)
        rate = rate * mean / overdispersion
        return jax.random.poisson(poisson_key, lam=rate, shape=shape)
    if isinstance(distribution, MultivariateNormal):
        mean, scale_tril = _mvn_mean_tril(distribution)
        validate_scale_tril(scale_tril, name="MultivariateNormal scale_tril")
        standard = jax.random.normal(
            key,
            shape=sample_shape + batch_shape(distribution) + event_shape(distribution),
        )
        shifted = jnp.einsum("...ij,...j->...i", jnp.tril(scale_tril), standard)
        return mean + shifted
    if isinstance(distribution, OrderedLogistic):
        _, cutpoints = _ordered_logistic_eta_cutpoints(distribution)
        if cutpoints.ndim == 0:
            raise ValueError("OrderedLogistic cutpoints must be a vector")
        ordered = jnp.all(cutpoints[..., 1:] > cutpoints[..., :-1])
        try:
            if not bool(ordered):
                raise ValueError("OrderedLogistic cutpoints must be strictly increasing")
        except jax.errors.TracerBoolConversionError:
            pass
        probabilities = _ordered_logistic_category_probabilities(distribution)
        dtype = jnp.result_type(probabilities, 1.0)
        safe_probabilities = jnp.clip(probabilities, jnp.finfo(dtype).tiny, 1.0)
        logits = jnp.log(jnp.broadcast_to(safe_probabilities, sample_shape + probabilities.shape))
        return jax.random.categorical(key, logits=logits, axis=-1).astype(jnp.int32)
    if isinstance(distribution, PythonSampleableDistribution):
        return cast(jax.Array, distribution.sample(key, sample_shape=sample_shape))
    raise TypeError(f"Distribution has no JAX sample support: {type(distribution).__name__}")


def cdf(distribution: Distribution, x: DistributionValue) -> jax.Array:
    """Return cumulative probability with the JAX backend."""
    if isinstance(distribution, Normal):
        loc, scale = _normal_loc_scale(distribution)
        return ndtr((jnp.asarray(x) - loc) / scale)
    if isinstance(distribution, HalfNormal):
        value = jnp.asarray(x)
        probability = 2.0 * ndtr(value / _halfnormal_scale(distribution)) - 1.0
        return jnp.where(value >= 0.0, probability, 0.0)
    if isinstance(distribution, Exponential):
        rate = _exponential_rate(distribution)
        value = jnp.asarray(x)
        return jnp.where(value >= 0.0, -jnp.expm1(-rate * value), 0.0)
    if isinstance(distribution, Uniform):
        low, high = _uniform_low_high(distribution)
        standardized = (jnp.asarray(x) - low) / (high - low)
        return jnp.clip(standardized, 0.0, 1.0)
    if isinstance(distribution, Truncated):
        _validate_scalar_event_truncated(distribution)
        if isinstance(distribution.base, Normal):
            return _normal_truncated_cdf(distribution, x)
        lower_probability, upper_probability = _truncated_probability_bounds(distribution)
        normalizer = upper_probability - lower_probability
        safe_normalizer = jnp.where(normalizer > 0.0, normalizer, 1.0)
        standardized = (cdf(distribution.base, x) - lower_probability) / safe_normalizer
        return jnp.clip(standardized, 0.0, 1.0)
    if isinstance(distribution, PythonInverseCdfDistribution):
        return cast(jax.Array, distribution.cdf(x))
    raise TypeError(f"Distribution has no JAX cdf support: {type(distribution).__name__}")


def icdf(distribution: Distribution, p: DistributionValue) -> jax.Array:
    """Return inverse cumulative probability with the JAX backend."""
    if isinstance(distribution, Normal):
        loc, scale = _normal_loc_scale(distribution)
        return loc + scale * ndtri(jnp.asarray(p))
    if isinstance(distribution, HalfNormal):
        return _halfnormal_scale(distribution) * ndtri(0.5 * (jnp.asarray(p) + 1.0))
    if isinstance(distribution, Exponential):
        return -jnp.log1p(-jnp.asarray(p)) / _exponential_rate(distribution)
    if isinstance(distribution, Uniform):
        low, high = _uniform_low_high(distribution)
        return low + (high - low) * jnp.asarray(p)
    if isinstance(distribution, Truncated):
        _validate_scalar_event_truncated(distribution)
        if isinstance(distribution.base, Normal):
            return _normal_truncated_icdf(distribution, p)
        lower_probability, upper_probability = _truncated_probability_bounds(distribution)
        base_probability = lower_probability + jnp.asarray(p) * (
            upper_probability - lower_probability
        )
        return icdf(distribution.base, base_probability)
    if isinstance(distribution, PythonInverseCdfDistribution):
        return cast(jax.Array, distribution.icdf(p))
    raise TypeError(f"Distribution has no JAX icdf support: {type(distribution).__name__}")
