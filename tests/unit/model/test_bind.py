"""Unit tests for binding helpers and parameter shape resolution."""

from __future__ import annotations

from typing import Protocol, cast

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedObserved,
    ResolvedParam,
    make_bind,
    param_count,
    resolve_param_shape,
)
from jaxstanv5.model.expr import DataRef


class BindFn(Protocol):
    def __call__(self, _cls: type[object], **values: object) -> object: ...


def make_meta() -> ModelMeta:
    return ModelMeta(
        params={
            "alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
            "beta": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=2),
            "group": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=DataRef("n")),
        },
        data_slots=["x", "n"],
        observed_name="y",
        observed=ResolvedObserved(Normal(0.0, 1.0)),
        expressions={},
    )


def bind_meta(meta: ModelMeta, **values: object) -> BoundModel:
    bind = cast(BindFn, make_bind(meta))
    return cast(BoundModel, bind(type("Model", (), {}), **values))


def test_bind_rejects_missing_data_and_observed_values() -> None:
    with pytest.raises(ValueError, match=r"Missing model data: \['n', 'y'\]"):
        bind_meta(make_meta(), x=jnp.asarray([1.0]))


def test_bind_rejects_extra_values() -> None:
    with pytest.raises(ValueError, match=r"Unexpected model data: \['z'\]"):
        bind_meta(
            make_meta(),
            x=jnp.asarray([1.0]),
            n=3,
            y=jnp.asarray([2.0]),
            z=0,
        )


def test_bind_converts_data_to_arrays_and_resolves_all_parameter_shapes() -> None:
    bound = bind_meta(make_meta(), x=[1.0, 2.0], n=3, y=[0.0, 1.0])

    assert isinstance(bound.data["x"], jax.Array)
    assert isinstance(bound.data["n"], jax.Array)
    assert bound.param_shapes == {"alpha": (), "beta": (2,), "group": (3,)}
    assert bound.n_params == 6


def test_resolve_param_shape_rejects_negative_literal_size() -> None:
    with pytest.raises(ValueError, match="Parameter size must be non-negative"):
        resolve_param_shape(-1, {})


def test_resolve_param_shape_rejects_bool_literal_size() -> None:
    with pytest.raises(TypeError, match="Parameter size must be an integer, not bool"):
        resolve_param_shape(True, {})


def test_resolve_param_shape_rejects_non_scalar_data_size() -> None:
    with pytest.raises(ValueError, match="Data-dependent parameter size 'n' must be scalar"):
        resolve_param_shape(DataRef("n"), {"n": jnp.asarray([2, 3])})


def test_resolve_param_shape_rejects_float_data_size() -> None:
    with pytest.raises(TypeError, match="Data-dependent parameter size 'n' must be integer"):
        resolve_param_shape(DataRef("n"), {"n": jnp.asarray(2.5)})


def test_resolve_param_shape_rejects_negative_data_size() -> None:
    with pytest.raises(ValueError, match="Data-dependent parameter size 'n' must be non-negative"):
        resolve_param_shape(DataRef("n"), {"n": jnp.asarray(-1)})


def test_param_count_multiplies_dimensions_and_rejects_negative_dimensions() -> None:
    assert param_count(()) == 1
    assert param_count((2, 3, 4)) == 24

    with pytest.raises(ValueError, match="Parameter shape dimensions must be non-negative"):
        param_count((2, -1))
