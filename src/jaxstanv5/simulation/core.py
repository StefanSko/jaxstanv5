"""Prior and prior-predictive simulation internals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.distributions.normal import Normal


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


def simulate_prior_predictive(
    model_cls: object,
    *,
    seed: int,
    num_samples: int,
    data: Mapping[str, object] | None = None,
    observed_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> PriorPredictiveResult:
    """Draw from a model's prior and prior predictive distribution."""
    raise NotImplementedError("Prior-predictive simulation is not implemented yet")
