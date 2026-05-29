"""Tests for prior and prior-predictive simulation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal
from jaxstanv5.simulation import simulate_prior_predictive


@model
class PriorOnlyNormal:
    """Prior-only scalar model."""

    mu = Param(Normal(0.0, 1.0))


@model
class NormalWithObserved:
    """Scalar prior with vector observed predictive draws."""

    mu = Param(Normal(0.0, 1.0))
    y = Observed(Normal(mu, 1.0))


@model
class PositiveScalePrior:
    """Positive-constrained prior-only model."""

    sigma = Param(Normal(0.0, 1.0), constraint=Positive())


@model
class DataSizedPrior:
    """Data-dependent vector prior."""

    n = Data()
    theta = Param(Normal(0.0, 1.0), size=n)


def test_simulate_prior_predictive_draws_prior_only_parameters_with_vmap_shape() -> None:
    result = simulate_prior_predictive(PriorOnlyNormal, seed=1, num_samples=7)

    assert result.parameters["mu"].shape == (7,)
    assert result.observed == {}
    assert result.data == {}
    assert jnp.all(jnp.isfinite(result.parameters["mu"]))


def test_simulate_prior_predictive_draws_observed_values_with_requested_shape() -> None:
    result = simulate_prior_predictive(
        NormalWithObserved,
        seed=2,
        num_samples=5,
        observed_shapes={"y": (3,)},
    )

    assert result.parameters["mu"].shape == (5,)
    assert result.observed["y"].shape == (5, 3)
    assert jnp.all(jnp.isfinite(result.observed["y"]))


def test_simulate_prior_predictive_draws_positive_constrained_parameters() -> None:
    result = simulate_prior_predictive(PositiveScalePrior, seed=3, num_samples=9)

    assert result.parameters["sigma"].shape == (9,)
    assert jnp.all(result.parameters["sigma"] > 0.0)


def test_simulate_prior_predictive_resolves_data_dependent_parameter_shape() -> None:
    result = simulate_prior_predictive(DataSizedPrior, seed=4, num_samples=6, data={"n": 4})

    assert result.parameters["theta"].shape == (6, 4)
    assert result.data["n"].shape == ()


def test_simulate_prior_predictive_rejects_missing_data() -> None:
    with pytest.raises(ValueError, match="Missing model data"):
        simulate_prior_predictive(DataSizedPrior, seed=5, num_samples=2)


def test_simulate_prior_predictive_rejects_unknown_observed_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected observed shapes"):
        simulate_prior_predictive(
            PriorOnlyNormal,
            seed=6,
            num_samples=2,
            observed_shapes={"y": (1,)},
        )
