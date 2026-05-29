"""Tests for constraint-aware prior sampling primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5.constraints import Positive
from jaxstanv5.constraints.core import ConstrainedValue, LogAbsDetJacobian, UnconstrainedValue
from jaxstanv5.distributions import Normal
from jaxstanv5.simulation.core import _sample_prior_value


class UnsupportedConstraint:
    """Constraint shape used to verify explicit unsupported-state failures."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        return x

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        return y

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        return jnp.zeros_like(jnp.asarray(y))


def test_sample_prior_value_draws_unconstrained_normal_with_requested_shape() -> None:
    key = jax.random.PRNGKey(11)

    value = _sample_prior_value(key, Normal(1.0, 2.0), constraint=None, sample_shape=(5,))

    assert value.shape == (5,)
    assert jnp.all(jnp.isfinite(value))


def test_sample_prior_value_draws_positive_truncated_normal_for_positive_constraint() -> None:
    key = jax.random.PRNGKey(12)

    value = _sample_prior_value(key, Normal(0.0, 1.0), constraint=Positive(), sample_shape=(100,))

    assert value.shape == (100,)
    assert jnp.all(value > 0.0)


def test_sample_prior_value_rejects_unsupported_constraint() -> None:
    with pytest.raises(TypeError, match="Unsupported prior constraint"):
        _sample_prior_value(
            jax.random.PRNGKey(13),
            Normal(0.0, 1.0),
            constraint=UnsupportedConstraint(),
            sample_shape=(),
        )
