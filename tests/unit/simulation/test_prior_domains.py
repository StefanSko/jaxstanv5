"""Tests for prior simulation domains."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5.constraints import Interval, Positive, UnitInterval
from jaxstanv5.constraints.core import (
    ConstrainedValue,
    LogAbsDetJacobian,
    UnconstrainedValue,
)
from jaxstanv5.simulation.domains import (
    ScalarIntervalDomain,
    UnconstrainedDomain,
    prior_domain_for_constraint,
)


class UnsupportedConstraint:
    """Constraint without prior simulation domain support."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        return jnp.asarray(x)

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        return jnp.asarray(y)

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        return jnp.zeros_like(jnp.asarray(y))


def test_prior_domain_for_no_constraint_is_unconstrained() -> None:
    assert prior_domain_for_constraint(None) == UnconstrainedDomain()


def test_prior_domain_for_positive_is_lower_bounded_interval() -> None:
    assert prior_domain_for_constraint(Positive()) == ScalarIntervalDomain(
        lower=0.0,
        upper=None,
    )


def test_prior_domain_for_interval_is_bounded_interval() -> None:
    assert prior_domain_for_constraint(Interval(-2.0, 3.0)) == ScalarIntervalDomain(
        lower=-2.0,
        upper=3.0,
    )


def test_prior_domain_for_unit_interval_is_bounded_interval() -> None:
    assert prior_domain_for_constraint(UnitInterval()) == ScalarIntervalDomain(
        lower=0.0,
        upper=1.0,
    )


def test_prior_domain_for_unsupported_constraint_raises() -> None:
    with pytest.raises(TypeError, match="Unsupported prior constraint"):
        prior_domain_for_constraint(UnsupportedConstraint())
