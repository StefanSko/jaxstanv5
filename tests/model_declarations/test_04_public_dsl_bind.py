"""Phase 4: public DSL and binding smoke tests.

This is the user-facing end-to-end path for the model-declaration slice:
``@model`` class -> final metadata -> ``bind(...)`` -> ``BoundModel``.
"""

from typing import Protocol, cast

import jax.numpy as jnp

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.core import DistributionParameter
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model import BoundModel, ModelMeta


class DecoratedModelLike(Protocol):
    _model_meta: ModelMeta

    def bind(self, **values: object) -> BoundModel: ...


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


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


def test_model_decorator_collects_declarations() -> None:
    linear_regression = cast(DecoratedModelLike, LinearRegression)
    meta = linear_regression._model_meta

    assert list(meta.params) == ["alpha", "beta", "sigma"]
    assert meta.data_slots == ["x"]
    assert meta.observed_name == "y"
    assert set(meta.expressions) == {"mu"}


def test_bind_attaches_data_and_resolves_scalar_parameter_shapes() -> None:
    linear_regression = cast(DecoratedModelLike, LinearRegression)
    bound = linear_regression.bind(x=jnp.asarray([1.0]), y=jnp.asarray([2.0]))

    assert bound.param_shapes == {"alpha": (), "beta": (), "sigma": ()}
    assert bound.n_params == 3


def test_bind_resolves_data_dependent_hierarchical_parameter_sizes() -> None:
    hierarchical_regression = cast(DecoratedModelLike, HierarchicalRegression)
    bound = hierarchical_regression.bind(
        n_groups=3,
        y=jnp.asarray([0.0, 1.0, 2.0]),
    )

    assert bound.param_shapes["alpha_pop"] == ()
    assert bound.param_shapes["sigma_alpha"] == ()
    assert bound.param_shapes["alpha"] == (3,)
    assert bound.n_params == 5
