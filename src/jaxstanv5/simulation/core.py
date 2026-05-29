"""Prior and prior-predictive simulation internals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from jaxstanv5.compiler.core import _evaluate_distribution
from jaxstanv5.constraints.core import Constraint
from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model.decorator import ModelMeta, _resolve_param_shape


class _ModelWithMeta(Protocol):
    """Decorated model class with attached resolved metadata."""

    _model_meta: ModelMeta


@dataclass(frozen=True)
class PriorPredictiveResult:
    """Draws from a model's prior and prior-predictive distribution."""

    parameters: Mapping[str, jax.Array]
    observed: Mapping[str, jax.Array]
    data: Mapping[str, jax.Array]


def _sample_normal(
    key: jax.Array,
    distribution: Normal,
    *,
    sample_shape: tuple[int, ...],
) -> jax.Array:
    return distribution.sample(key, sample_shape=sample_shape)


def _sample_positive_normal(
    key: jax.Array,
    distribution: Normal,
    *,
    sample_shape: tuple[int, ...],
) -> jax.Array:
    loc = jnp.asarray(distribution.loc)
    scale = jnp.asarray(distribution.scale)
    event_shape = jnp.broadcast_shapes(loc.shape, scale.shape)
    lower_probability = ndtr((0.0 - loc) / scale)
    uniform = jax.random.uniform(
        key,
        shape=sample_shape + event_shape,
        minval=lower_probability,
        maxval=1.0,
    )
    return loc + scale * ndtri(uniform)


def _sample_prior_value(
    key: jax.Array,
    distribution: Distribution,
    *,
    constraint: Constraint | None,
    sample_shape: tuple[int, ...],
) -> jax.Array:
    """Sample one parameter value from its constrained-space prior."""
    if constraint is None:
        if isinstance(distribution, Normal):
            return _sample_normal(key, distribution, sample_shape=sample_shape)
        raise TypeError(f"Unsupported prior distribution: {type(distribution).__name__}")

    if isinstance(constraint, Positive):
        if isinstance(distribution, Normal):
            return _sample_positive_normal(key, distribution, sample_shape=sample_shape)
        raise TypeError(
            f"Unsupported positive-constrained prior distribution: {type(distribution).__name__}"
        )

    raise TypeError(f"Unsupported prior constraint: {type(constraint).__name__}")


def _model_meta(model_cls: object) -> ModelMeta:
    if not hasattr(model_cls, "_model_meta"):
        raise TypeError("model_cls must be decorated with @model")
    return cast(_ModelWithMeta, model_cls)._model_meta


def _normalize_data(meta: ModelMeta, data: Mapping[str, object] | None) -> dict[str, jax.Array]:
    raw_data: Mapping[str, object] = {} if data is None else data
    expected = set(meta.data_slots)
    actual = set(raw_data)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing model data: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected model data: {sorted(extra)}")
    return {name: jnp.asarray(value) for name, value in raw_data.items()}


def _resolve_param_shapes(
    meta: ModelMeta, data: dict[str, jax.Array]
) -> dict[str, tuple[int, ...]]:
    return {name: _resolve_param_shape(param.size, data) for name, param in meta.params.items()}


def _validate_observed_shapes(
    meta: ModelMeta,
    observed_shapes: Mapping[str, tuple[int, ...]] | None,
) -> dict[str, tuple[int, ...]]:
    raw_shapes: Mapping[str, tuple[int, ...]] = {} if observed_shapes is None else observed_shapes
    observed_names = {observed.name for observed in meta.observed_nodes}
    extra = set(raw_shapes) - observed_names
    if extra:
        raise ValueError(f"Unexpected observed shapes: {sorted(extra)}")

    result: dict[str, tuple[int, ...]] = {}
    for observed in meta.observed_nodes:
        shape = raw_shapes.get(observed.name, ())
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
    observed_shapes: dict[str, tuple[int, ...]],
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
            sample_shape=param_shapes[name],
        )
        parameters[name] = value
        values[name] = value
        key_index += 1

    observed_values: dict[str, jax.Array] = {}
    for observed in meta.observed_nodes:
        distribution = _evaluate_distribution(observed.distribution, values)
        observed_value = _sample_prior_value(
            keys[key_index],
            distribution,
            constraint=None,
            sample_shape=observed_shapes[observed.name],
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
    normalized_data = _normalize_data(meta, data)
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
