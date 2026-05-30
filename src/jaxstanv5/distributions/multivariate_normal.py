"""Multivariate Normal distribution."""

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


@dataclass(frozen=True)
class MultivariateNormal:
    """Event-wise multivariate Normal with a lower Cholesky scale factor.

    This distribution intentionally exposes only ``log_prob`` for now. Prior
    simulation needs explicit event-shape support before MVN sampling can be
    wired safely.
    """

    mean: DistributionParameter
    scale_tril: DistributionParameter

    def _mean_tril(self) -> tuple[jax.Array, jax.Array]:
        return (
            jnp.asarray(_concrete_parameter(self.mean)),
            jnp.asarray(_concrete_parameter(self.scale_tril)),
        )

    def event_shape(self) -> tuple[int, ...]:
        """Return the vector event shape."""
        _, scale_tril = self._mean_tril()
        return (scale_tril.shape[-1],)

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return event-wise multivariate Normal log-density for ``x``."""
        mean, scale_tril = self._mean_tril()
        value = jnp.asarray(x)
        delta = value - mean
        event_size = scale_tril.shape[-1]
        flat_delta = delta.reshape((-1, event_size))
        solved = jax.vmap(lambda row: solve_triangular(scale_tril, row, lower=True))(flat_delta)
        solved = solved.reshape(delta.shape)
        quadratic = jnp.sum(solved**2, axis=-1)
        log_det = jnp.sum(jnp.log(jnp.diagonal(scale_tril, axis1=-2, axis2=-1)))
        log_prob = -0.5 * quadratic - log_det - 0.5 * event_size * math.log(2.0 * math.pi)
        return jnp.reshape(log_prob, delta.shape[:-1])
