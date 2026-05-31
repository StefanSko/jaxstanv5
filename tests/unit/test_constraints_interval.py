"""Tests for finite interval constraints."""

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5.constraints import Interval, UnitInterval


def test_interval_rejects_bool_bounds() -> None:
    with pytest.raises(TypeError, match="not bool"):
        Interval(False, 1.0)

    with pytest.raises(TypeError, match="not bool"):
        Interval(0.0, True)


def test_interval_rejects_non_finite_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        Interval(float("-inf"), 1.0)

    with pytest.raises(ValueError, match="finite"):
        Interval(0.0, float("nan"))


def test_interval_rejects_unordered_bounds() -> None:
    with pytest.raises(ValueError, match="less than"):
        Interval(1.0, 1.0)

    with pytest.raises(ValueError, match="less than"):
        Interval(2.0, 1.0)


def test_interval_constraint_round_trips_constrained_values() -> None:
    constraint = Interval(-2.0, 3.0)
    constrained = jnp.asarray([-1.75, -0.5, 2.5])

    unconstrained = constraint.transform(constrained)
    actual = constraint.inverse_transform(unconstrained)

    assert jnp.allclose(actual, constrained)


def test_interval_inverse_transform_lands_inside_interval() -> None:
    constraint = Interval(-2.0, 3.0)
    unconstrained = jnp.asarray([-10.0, -2.0, 0.0, 2.0, 10.0])

    actual = jnp.asarray(constraint.inverse_transform(unconstrained))

    assert jnp.all(actual > -2.0)
    assert jnp.all(actual < 3.0)


def test_interval_log_abs_det_jacobian_matches_scaled_logit_transform() -> None:
    constraint = Interval(-2.0, 3.0)
    unconstrained = jnp.asarray([-1.0, 0.0, 2.0])

    actual = constraint.log_abs_det_jacobian(unconstrained)

    expected = jnp.log(5.0) - jax.nn.softplus(-unconstrained) - jax.nn.softplus(unconstrained)
    assert jnp.allclose(actual, expected)


def test_unit_interval_matches_interval_zero_one() -> None:
    unit = UnitInterval()
    interval = Interval(0.0, 1.0)
    unconstrained = jnp.asarray([-2.0, 0.0, 3.0])
    constrained = jnp.asarray([0.2, 0.5, 0.8])

    assert jnp.allclose(unit.transform(constrained), interval.transform(constrained))
    assert jnp.allclose(
        unit.inverse_transform(unconstrained), interval.inverse_transform(unconstrained)
    )
    assert jnp.allclose(
        unit.log_abs_det_jacobian(unconstrained),
        interval.log_abs_det_jacobian(unconstrained),
    )
