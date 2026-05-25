"""Public DSL binding smoke tests.

This is the user-facing model-declaration path:
``@model`` class -> ``bind(...)`` -> returned bound model object.
"""

from __future__ import annotations

from collections.abc import Callable
from operator import attrgetter
from typing import cast

import jax.numpy as jnp

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import DistributionParameter, Normal


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


def bind_model(model_cls: object, **values: object) -> object:
    bind = attrgetter("bind")(model_cls)
    assert callable(bind)
    return cast(Callable[..., object], bind)(**values)


def bound_param_shapes(bound: object) -> dict[str, tuple[int, ...]]:
    param_shapes = attrgetter("param_shapes")(bound)
    assert isinstance(param_shapes, dict)
    for name, shape in param_shapes.items():
        assert isinstance(name, str)
        assert isinstance(shape, tuple)
        for dim in shape:
            assert isinstance(dim, int)
    return cast(dict[str, tuple[int, ...]], param_shapes)


def bound_n_params(bound: object) -> int:
    n_params = attrgetter("n_params")(bound)
    assert isinstance(n_params, int)
    return n_params


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(dist_param(mu), dist_param(sigma)))


@model
class HierarchicalRegression:
    n_groups = Data()
    alpha_pop = Param(Normal(0.0, 1.0))
    sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
    alpha = Param(Normal(dist_param(alpha_pop), dist_param(sigma_alpha)), size=n_groups)
    y = Observed(Normal(dist_param(alpha), 1.0))


def test_bind_resolves_scalar_parameter_shapes() -> None:
    bound = bind_model(LinearRegression, x=jnp.asarray([1.0]), y=jnp.asarray([2.0]))

    assert bound_param_shapes(bound) == {"alpha": (), "beta": (), "sigma": ()}
    assert bound_n_params(bound) == 3


def test_bind_resolves_data_dependent_hierarchical_parameter_sizes() -> None:
    bound = bind_model(
        HierarchicalRegression,
        n_groups=3,
        y=jnp.asarray([0.0, 1.0, 2.0]),
    )

    assert bound_param_shapes(bound) == {
        "alpha_pop": (),
        "sigma_alpha": (),
        "alpha": (3,),
    }
    assert bound_n_params(bound) == 5


def test_bind_accepts_multiple_observed_likelihood_sites() -> None:
    @model
    class MeasurementErrorModel:
        n = Data()
        x_sd = Data()
        y_sd = Data()

        alpha = Param(Normal(0.0, 1.0))
        beta = Param(Normal(0.0, 1.0))
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())

        x_true = Param(Normal(0.0, 1.0), size=n)
        mu = alpha + beta * x_true
        y_true = Param(Normal(dist_param(mu), dist_param(sigma)), size=n)

        x_obs = Observed(Normal(dist_param(x_true), dist_param(x_sd)))
        y_obs = Observed(Normal(dist_param(y_true), dist_param(y_sd)))

    bound = bind_model(
        MeasurementErrorModel,
        n=3,
        x_sd=jnp.asarray([0.05, 0.05, 0.05]),
        y_sd=jnp.asarray([0.10, 0.10, 0.10]),
        x_obs=jnp.asarray([-1.0, 0.0, 1.0]),
        y_obs=jnp.asarray([0.5, 1.0, 1.5]),
    )

    assert bound_param_shapes(bound) == {
        "alpha": (),
        "beta": (),
        "sigma": (),
        "x_true": (3,),
        "y_true": (3,),
    }
    assert bound_n_params(bound) == 9
