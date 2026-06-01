"""Multivariate probability distributions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from jaxstanv5.distributions.core import (
    DistributionParameter,
    DistributionValue,
    LogProbability,
    _concrete_parameter,
)


def _event_value(value: jax.Array, event_size: int) -> jax.Array:
    """Return ``value`` with an explicit trailing event dimension."""
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


@dataclass(frozen=True)
class MultivariateNormal:
    """Event-wise multivariate Normal with a lower Cholesky scale factor.

    Samples have shape ``sample_shape + batch_shape + event_shape``.
    """

    mean: DistributionParameter
    scale_tril: DistributionParameter

    def _mean_tril(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.mean)),
            jnp.asarray(_concrete_parameter(self.scale_tril)),
        )

    def batch_shape(self) -> tuple[int, ...]:
        """Return broadcasted non-event dimensions for MVN parameters."""
        mean, scale_tril = self._mean_tril()
        mean_batch_shape = () if mean.ndim == 0 else mean.shape[:-1]
        scale_batch_shape = scale_tril.shape[:-2]
        return jnp.broadcast_shapes(mean_batch_shape, scale_batch_shape)

    def event_shape(self) -> tuple[int, ...]:
        """Return the vector event shape."""
        _, scale_tril = self._mean_tril()
        return (scale_tril.shape[-1],)

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        """Draw MVN samples with leading ``sample_shape`` dimensions."""
        mean, scale_tril = self._mean_tril()
        standard = jax.random.normal(
            key, shape=sample_shape + self.batch_shape() + self.event_shape()
        )
        shifted = jnp.einsum("...ij,...j->...i", scale_tril, standard)
        return mean + shifted

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return event-wise multivariate Normal log-density for ``x``."""
        mean, scale_tril = self._mean_tril()
        event_size = scale_tril.shape[-1]
        value = _event_value(jnp.asarray(x), event_size)
        delta = value - mean
        batch_shape = jnp.broadcast_shapes(delta.shape[:-1], scale_tril.shape[:-2])

        delta = jnp.broadcast_to(delta, batch_shape + (event_size,))
        scale_tril = jnp.broadcast_to(scale_tril, batch_shape + (event_size, event_size))

        flat_delta = delta.reshape((-1, event_size))
        flat_scale_tril = scale_tril.reshape((-1, event_size, event_size))
        solved = jax.vmap(lambda tril, row: solve_triangular(tril, row, lower=True))(
            flat_scale_tril,
            flat_delta,
        )
        solved = solved.reshape(batch_shape + (event_size,))
        quadratic = jnp.sum(solved**2, axis=-1)
        log_det = jnp.sum(
            jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)),
            axis=-1,
        )
        log_prob = -0.5 * quadratic - log_det - 0.5 * event_size * math.log(2.0 * math.pi)
        return jnp.reshape(log_prob, batch_shape)
