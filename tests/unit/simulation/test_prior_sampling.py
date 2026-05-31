"""Tests for constraint-aware prior sampling primitives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5.constraints import Positive
from jaxstanv5.constraints.core import ConstrainedValue, LogAbsDetJacobian, UnconstrainedValue
from jaxstanv5.distributions import Normal
from jaxstanv5.distributions.core import DistributionValue, LogProbability
from jaxstanv5.simulation.core import _leading_sample_shape, _sample_prior_value


class UnitIntervalDistribution:
    """Simple inverse-CDF distribution for generic constrained sampling tests."""

    def batch_shape(self) -> tuple[int, ...]:
        return ()

    def event_shape(self) -> tuple[int, ...]:
        return ()

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return jnp.zeros_like(jnp.asarray(x))

    def sample(
        self,
        key: jax.Array,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> jax.Array:
        return jax.random.uniform(key, shape=sample_shape)

    def cdf(self, x: DistributionValue) -> jax.Array:
        return jnp.asarray(x)

    def icdf(self, p: DistributionValue) -> jax.Array:
        return jnp.asarray(p)


class UnsupportedConstraint:
    """Constraint shape used to verify explicit unsupported-state failures."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        return x

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        return y

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        return jnp.zeros_like(jnp.asarray(y))


def test_leading_sample_shape_keeps_scalar_event_vector_shape() -> None:
    sample_shape = _leading_sample_shape(
        target_shape=(3,),
        batch_shape=(),
        event_shape=(),
    )

    assert sample_shape == (3,)


def test_leading_sample_shape_removes_event_shape_suffix() -> None:
    sample_shape = _leading_sample_shape(
        target_shape=(3,),
        batch_shape=(),
        event_shape=(3,),
    )

    assert sample_shape == ()


def test_leading_sample_shape_removes_batch_shape_suffix() -> None:
    sample_shape = _leading_sample_shape(
        target_shape=(2, 3),
        batch_shape=(3,),
        event_shape=(),
    )

    assert sample_shape == (2,)


def test_leading_sample_shape_rejects_incompatible_event_suffix() -> None:
    with pytest.raises(ValueError, match="Target shape"):
        _leading_sample_shape(
            target_shape=(2,),
            batch_shape=(),
            event_shape=(3,),
        )


def test_sample_prior_value_draws_unconstrained_normal_with_requested_shape() -> None:
    key = jax.random.PRNGKey(11)

    value = _sample_prior_value(key, Normal(1.0, 2.0), constraint=None, target_shape=(5,))

    assert value.shape == (5,)
    assert jnp.all(jnp.isfinite(value))


def test_sample_prior_value_draws_positive_truncated_normal_for_positive_constraint() -> None:
    key = jax.random.PRNGKey(12)

    value = _sample_prior_value(key, Normal(0.0, 1.0), constraint=Positive(), target_shape=(100,))

    assert value.shape == (100,)
    assert jnp.all(value > 0.0)


def test_sample_prior_value_uses_inverse_cdf_distribution_for_positive_constraint() -> None:
    value = _sample_prior_value(
        jax.random.PRNGKey(14),
        UnitIntervalDistribution(),
        constraint=Positive(),
        target_shape=(20,),
    )

    assert value.shape == (20,)
    assert jnp.all(value >= 0.0)
    assert jnp.all(value <= 1.0)


def test_sample_prior_value_rejects_unsupported_constraint() -> None:
    with pytest.raises(TypeError, match="Unsupported prior constraint"):
        _sample_prior_value(
            jax.random.PRNGKey(13),
            Normal(0.0, 1.0),
            constraint=UnsupportedConstraint(),
            target_shape=(),
        )
