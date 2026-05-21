"""Tests for declarative model classes and binding."""

import jax.numpy as jnp

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.normal import Normal


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


@model
class HierarchicalRegression:
    n_groups = Data()
    alpha_pop = Param(Normal(0.0, 1.0))
    sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    y = Observed(Normal(alpha, 1.0))


def test_model_decorator_collects_declarations() -> None:
    meta = LinearRegression._model_meta

    assert list(meta.params) == ["alpha", "beta", "sigma"]
    assert meta.data_slots == ["x"]
    assert meta.observed_name == "y"
    assert set(meta.expressions) == {"mu"}


def test_bind_attaches_data_and_resolves_scalar_parameter_shapes() -> None:
    bound = LinearRegression.bind(x=jnp.asarray([1.0]), y=jnp.asarray([2.0]))

    assert bound.param_shapes == {"alpha": (), "beta": (), "sigma": ()}
    assert bound.n_params == 3


def test_bind_resolves_data_dependent_hierarchical_parameter_sizes() -> None:
    bound = HierarchicalRegression.bind(
        n_groups=3,
        y=jnp.asarray([0.0, 1.0, 2.0]),
    )

    assert bound.param_shapes["alpha_pop"] == ()
    assert bound.param_shapes["sigma_alpha"] == ()
    assert bound.param_shapes["alpha"] == (3,)
    assert bound.n_params == 5
