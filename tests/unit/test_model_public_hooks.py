from __future__ import annotations

from typing import Protocol, cast

import pytest

from jaxstanv5.distributions import Normal
from jaxstanv5.ir import bindable_from_meta
from jaxstanv5.model import (
    BoundModel,
    Dim,
    ModelMeta,
    Observed,
    Param,
    bind_model,
    dimension_metadata_to_dict,
    is_model_class,
    model,
    model_meta,
)


class _Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


obs = Dim("obs", coords=("a", "b"))


@model
class PublicHookModel:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0), dims=(obs,))


def test_model_meta_returns_public_metadata_for_decorated_model() -> None:
    meta = model_meta(PublicHookModel)

    assert isinstance(meta, ModelMeta)
    assert tuple(meta.params) == ("theta",)
    assert tuple(observed.name for observed in meta.observed_nodes) == ("y",)


def test_is_model_class_identifies_decorated_model_classes() -> None:
    assert is_model_class(PublicHookModel)
    assert not is_model_class(object)
    assert not is_model_class(PublicHookModel())
    assert not is_model_class(object())


def test_model_meta_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        model_meta(object)

    with pytest.raises(TypeError, match="@model"):
        model_meta(PublicHookModel())


def test_bind_model_binds_data_and_preserves_dimension_metadata() -> None:
    bound = bind_model(PublicHookModel, {"y": [1.0, 2.0]})

    assert isinstance(bound, BoundModel)
    y = cast(_Shaped, bound.data["y"])
    assert bound.meta is model_meta(PublicHookModel)
    assert bound.param_shapes == {"theta": ()}
    assert tuple(y.shape) == (2,)
    assert bound.dimensions is not None
    assert dimension_metadata_to_dict(bound.dimensions) == {
        "dims": {"theta": [], "y": ["obs"]},
        "coords": {"obs": ["a", "b"]},
    }


def test_public_hooks_accept_ir_bindable_models() -> None:
    meta = model_meta(PublicHookModel)
    rebuilt = bindable_from_meta(meta)

    assert is_model_class(rebuilt)
    assert model_meta(rebuilt) is meta
    bound = bind_model(rebuilt, {"y": [1.0, 2.0]})
    assert bound.meta is meta
    assert bound.dimensions is None


def test_bind_model_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        bind_model(object, {})
