"""Prior and prior-predictive simulation internals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp

from jaxstanv5._backends.jax.distributions import (
    batch_shape,
    cdf,
    event_shape,
    icdf,
    is_inverse_cdf,
    is_sampleable,
)
from jaxstanv5._backends.jax.distributions import (
    sample as distribution_sample,
)
from jaxstanv5.compiler.core import _evaluate_distribution
from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model.decorator import (
    ModelMeta,
    _normalize_declared_data_values,
    _resolve_param_shape,
    _resolved_free_values,
    _validate_bound_distribution_parameters,
)
from jaxstanv5.simulation.domains import (
    OrderedVectorDomain,
    ScalarIntervalDomain,
    UnconstrainedDomain,
    prior_domain_for_constraint,
)


class _ModelWithMeta(Protocol):
    """Decorated model class with attached resolved metadata."""

    _model_meta: ModelMeta


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Draws from a model's prior and prior-predictive distribution."""

    parameters: Mapping[str, jax.Array]
    observed: Mapping[str, jax.Array]
    data: Mapping[str, jax.Array]


def _leading_sample_shape(
    *,
    target_shape: tuple[int, ...],
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Return iid sample dimensions from a full target value shape."""
    suffix = batch_shape + event_shape
    if suffix == ():
        return target_shape
    if len(target_shape) < len(suffix) or target_shape[-len(suffix) :] != suffix:
        raise ValueError(
            "Target shape must end with distribution batch_shape + event_shape: "
            f"target_shape={target_shape}, batch_shape={batch_shape}, event_shape={event_shape}"
        )
    return target_shape[: -len(suffix)]


def _sample_interval_restricted(
    key: jax.Array,
    distribution: Distribution,
    domain: ScalarIntervalDomain,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    if event_shape(distribution) != ():
        raise TypeError("Interval-constrained prior simulation requires scalar-event distributions")
    sample_shape = _leading_sample_shape(
        target_shape=target_shape,
        batch_shape=batch_shape(distribution),
        event_shape=event_shape(distribution),
    )
    lower_probability = (
        jnp.asarray(0.0) if domain.lower is None else cdf(distribution, domain.lower)
    )
    upper_probability = (
        jnp.asarray(1.0) if domain.upper is None else cdf(distribution, domain.upper)
    )
    uniform = jax.random.uniform(
        key,
        shape=sample_shape + batch_shape(distribution),
        minval=lower_probability,
        maxval=upper_probability,
    )
    return icdf(distribution, uniform)


def _sample_ordered_vector(
    key: jax.Array,
    distribution: Distribution,
    *,
    target_shape: tuple[int, ...],
) -> jax.Array:
    """Sample an ordered vector from an iid scalar constrained-space prior."""
    if target_shape == ():
        raise ValueError("Ordered prior simulation requires vector target shape")
    if event_shape(distribution) != ():
        raise TypeError("Ordered prior simulation requires scalar-event distributions")
    if batch_shape(distribution) != ():
        raise TypeError("Ordered prior simulation requires iid scalar distributions")
    raw = distribution_sample(distribution, key, sample_shape=target_shape)
    return jnp.sort(raw, axis=-1)


def _sample_prior_value(
    key: jax.Array,
    distribution: Distribution,
    *,
    constraint: Constraint | None,
    target_shape: tuple[int, ...],
) -> jax.Array:
    """Sample one parameter value from its constrained-space prior."""
    domain = prior_domain_for_constraint(constraint)

    if isinstance(domain, UnconstrainedDomain):
        if is_sampleable(distribution):
            sample_shape = _leading_sample_shape(
                target_shape=target_shape,
                batch_shape=batch_shape(distribution),
                event_shape=event_shape(distribution),
            )
            return distribution_sample(distribution, key, sample_shape=sample_shape)
        raise TypeError(f"Unsupported prior distribution: {type(distribution).__name__}")

    if isinstance(domain, ScalarIntervalDomain):
        if is_inverse_cdf(distribution):
            return _sample_interval_restricted(
                key,
                distribution,
                domain,
                target_shape=target_shape,
            )
        raise TypeError(
            f"Unsupported interval-constrained prior distribution: {type(distribution).__name__}"
        )

    if isinstance(domain, OrderedVectorDomain):
        if is_sampleable(distribution):
            return _sample_ordered_vector(key, distribution, target_shape=target_shape)
        raise TypeError(f"Unsupported ordered prior distribution: {type(distribution).__name__}")

    raise TypeError(f"Unsupported prior domain: {type(domain).__name__}")


def _model_meta(model_cls: object) -> ModelMeta:
    if not hasattr(model_cls, "_model_meta"):
        raise TypeError("model_cls must be decorated with @model")
    return cast(_ModelWithMeta, model_cls)._model_meta


def _normalize_data(meta: ModelMeta, data: Mapping[str, object] | None) -> dict[str, jax.Array]:
    return _normalize_declared_data_values(meta, data)


def _resolve_param_shapes(
    meta: ModelMeta, data: dict[str, jax.Array]
) -> dict[str, tuple[int, ...]]:
    return {name: _resolve_param_shape(param.size, data) for name, param in meta.params.items()}


def _validate_observed_shapes(
    meta: ModelMeta,
    observed_shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...] | None]:
    raw_shapes: Mapping[str, tuple[int, ...]] = {} if observed_shapes is None else observed_shapes
    observed_names = {observed.name for observed in meta.observed_nodes}
    extra = set(raw_shapes) - observed_names
    if extra:
        raise ValueError(f"Unexpected observed shapes: {sorted(extra)}")

    result: dict[str, tuple[int, ...] | None] = {}
    for observed in meta.observed_nodes:
        shape = raw_shapes.get(observed.name)
        if shape is not None:
            for dim in shape:
                if isinstance(dim, bool):
                    raise TypeError("Observed shape dimensions must be integers, not bool")
                if dim < 0:
                    raise ValueError("Observed shape dimensions must be non-negative")
        result[observed.name] = shape
    return result


def _simulate_one(
    key: jax.Array,
    *,
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
    observed_shapes: dict[str, tuple[int, ...] | None],
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    keys = jax.random.split(key, len(meta.params) + len(meta.observed_nodes))
    key_index = 0
    parameters: dict[str, jax.Array] = {}
    values = dict(data)

    for name, param in meta.params.items():
        distribution = _evaluate_distribution(param.distribution, values)
        value = _sample_prior_value(
            keys[key_index],
            distribution,
            constraint=param.constraint,
            target_shape=param_shapes[name],
        )
        parameters[name] = value
        values[name] = value
        key_index += 1

    observed_values: dict[str, jax.Array] = {}
    for observed in meta.observed_nodes:
        distribution = _evaluate_distribution(observed.distribution, values)
        observed_target_shape = observed_shapes[observed.name]
        if observed_target_shape is None:
            if not is_sampleable(distribution):
                raise TypeError(f"Unsupported prior distribution: {type(distribution).__name__}")
            observed_target_shape = batch_shape(distribution) + event_shape(distribution)
        observed_value = _sample_prior_value(
            keys[key_index],
            distribution,
            constraint=None,
            target_shape=observed_target_shape,
        )
        observed_values[observed.name] = observed_value
        key_index += 1

    return parameters, observed_values


def simulate_prior_predictive(
    model_cls: object,
    *,
    seed: int,
    num_samples: int,
    data: Mapping[str, object] | None = None,
    observed_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> PriorPredictiveResult:
    """Draw from a model's prior and prior predictive distribution."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")

    meta = _model_meta(model_cls)
    non_param_free_values = set(_resolved_free_values(meta)) - set(meta.params)
    if non_param_free_values:
        raise TypeError(
            "PartiallyObserved declarations are not supported by prior-predictive simulation"
        )
    normalized_data = _normalize_data(meta, data)
    _validate_bound_distribution_parameters(meta, normalized_data)
    param_shapes = _resolve_param_shapes(meta, normalized_data)
    normalized_observed_shapes = _validate_observed_shapes(meta, observed_shapes)
    keys = jax.random.split(jax.random.PRNGKey(seed), num_samples)

    def draw_one(key: jax.Array) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        return _simulate_one(
            key,
            meta=meta,
            data=normalized_data,
            param_shapes=param_shapes,
            observed_shapes=normalized_observed_shapes,
        )

    parameters, observed = jax.jit(jax.vmap(draw_one))(keys)
    return PriorPredictiveResult(parameters=parameters, observed=observed, data=normalized_data)
