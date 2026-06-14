"""Unit tests for reconstructing a bindable model from decoded metadata."""

from __future__ import annotations

from typing import Protocol, cast

import jax.numpy as jnp
import pytest

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import Normal
from jaxstanv5.ir import bindable_from_meta, meta_from_dict, meta_to_dict
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


def declared_meta() -> ModelMeta:
    @model
    class LinearRegression:
        alpha = Param(Normal(0.0, 1.0))
        beta = Param(Normal(0.0, 1.0))
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())
        x = Data.vector()
        mu = alpha + beta * x
        y = Observed(Normal(mu, sigma))

    return cast(ModelMeta, getattr(LinearRegression, "_model_meta"))  # noqa: B009


def test_reconstructed_model_binds_like_the_decorated_class() -> None:
    meta = declared_meta()
    decoded = meta_from_dict(meta_to_dict(meta))

    rebuilt = cast(_BindableModel, bindable_from_meta(decoded))
    bound = rebuilt.bind(x=jnp.asarray([1.0, 2.0]), y=jnp.asarray([0.5, 1.5]))

    assert bound.meta == meta
    assert bound.param_shapes == {"alpha": (), "beta": (), "sigma": ()}
    assert bound.n_params == 3


def test_reconstructed_model_exposes_model_meta() -> None:
    meta = declared_meta()

    rebuilt = bindable_from_meta(meta)

    assert getattr(rebuilt, "_model_meta") is meta  # noqa: B009


def test_reconstructed_model_validates_bound_data() -> None:
    rebuilt = cast(_BindableModel, bindable_from_meta(declared_meta()))

    with pytest.raises(ValueError, match=r"Missing model data: \['y'\]"):
        rebuilt.bind(x=jnp.asarray([1.0, 2.0]))
