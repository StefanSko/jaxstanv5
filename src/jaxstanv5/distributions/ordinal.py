"""Ordinal probability distributions."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstanv5.distributions.core import (
    DiscreteDistribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


@dataclass(frozen=True)
class OrderedLogistic(DiscreteDistribution):
    """Ordered-logistic distribution with zero-based category labels.

    For ``K`` ordered cutpoints, valid observed labels are ``0, ..., K``.
    """

    eta: DistributionParameter
    cutpoints: DistributionParameter

    def _eta_cutpoints(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.eta)),
            jnp.asarray(_concrete_parameter(self.cutpoints)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-sample dimensions for observations."""
        eta, cutpoints = self._eta_cutpoints()
        if cutpoints.ndim == 0:
            raise ValueError("OrderedLogistic cutpoints must be a vector")
        return jnp.broadcast_shapes(eta.shape, cutpoints.shape[:-1])

    def event_shape(self) -> tuple[int, ...]:
        """Return scalar-event shape for element-wise ordinal observations."""
        return ()

    def _category_probabilities(self) -> jax.Array:
        eta, cutpoints = self._eta_cutpoints()
        if cutpoints.ndim == 0:
            raise ValueError("OrderedLogistic cutpoints must be a vector")
        if cutpoints.shape[-1] < 1:
            raise ValueError("OrderedLogistic requires at least one cutpoint")
        batch_shape = self.batch_shape()
        eta_broadcast = jnp.broadcast_to(eta, batch_shape)
        cutpoints_broadcast = jnp.broadcast_to(cutpoints, batch_shape + cutpoints.shape[-1:])
        cumulative = jax.nn.sigmoid(cutpoints_broadcast - eta_broadcast[..., None])
        first = cumulative[..., :1]
        middle = cumulative[..., 1:] - cumulative[..., :-1]
        last = 1.0 - cumulative[..., -1:]
        return jnp.concatenate((first, middle, last), axis=-1)

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise ordinal log-probability mass for ``x``."""
        eta, cutpoints = self._eta_cutpoints()
        probabilities = self._category_probabilities()
        category_count = cutpoints.shape[-1] + 1
        batch_shape = jnp.broadcast_shapes(probabilities.shape[:-1], jnp.asarray(x).shape)
        probabilities_broadcast = jnp.broadcast_to(probabilities, batch_shape + (category_count,))
        value = jnp.broadcast_to(jnp.asarray(x), batch_shape)
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
        ordered_broadcast = jnp.broadcast_to(ordered_broadcast, batch_shape)
        support = integer_value & in_range & ordered_broadcast & (selected_probability > 0.0)
        dtype = jnp.result_type(eta, cutpoints, 1.0)
        safe_probability = jnp.clip(selected_probability, jnp.finfo(dtype).tiny, 1.0)
        return jnp.where(support, jnp.log(safe_probability), -jnp.inf)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw zero-based ordinal category labels with leading sample dimensions."""
        _, cutpoints = self._eta_cutpoints()
        if cutpoints.ndim == 0:
            raise ValueError("OrderedLogistic cutpoints must be a vector")
        if not bool(jnp.all(cutpoints[..., 1:] > cutpoints[..., :-1])):
            raise ValueError("OrderedLogistic cutpoints must be strictly increasing")
        probabilities = self._category_probabilities()
        dtype = jnp.result_type(probabilities, 1.0)
        safe_probabilities = jnp.clip(probabilities, jnp.finfo(dtype).tiny, 1.0)
        logits = jnp.log(jnp.broadcast_to(safe_probabilities, sample_shape + probabilities.shape))
        return jax.random.categorical(key, logits=logits, axis=-1).astype(jnp.int32)
