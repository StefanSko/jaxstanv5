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
    dimension_metadata_from_dict,
    dimension_metadata_to_dict,
    is_model_class,
    model,
    model_dimensions,
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


def test_public_hooks_reject_subclasses_with_inherited_model_metadata() -> None:
    class ChildModel(PublicHookModel):
        extra = Param(Normal(0.0, 1.0))

    assert not is_model_class(ChildModel)
    with pytest.raises(TypeError, match="@model"):
        model_meta(ChildModel)
    with pytest.raises(TypeError, match="@model"):
        bind_model(ChildModel, {"y": [1.0, 2.0]})


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
    # Without the decoded dimension sidecar there is nothing to reattach.
    assert bound.dimensions is None


def test_ir_bindable_models_round_trip_dimension_metadata() -> None:
    meta = model_meta(PublicHookModel)
    declared_dimensions = model_dimensions(PublicHookModel)
    document = dimension_metadata_to_dict(declared_dimensions)
    rebuilt = bindable_from_meta(meta, dimensions=dimension_metadata_from_dict(document))

    assert is_model_class(rebuilt)
    assert model_dimensions(rebuilt) == declared_dimensions
    bound = bind_model(rebuilt, {"y": [1.0, 2.0]})
    assert bound.dimensions is not None
    assert bound.dimensions == declared_dimensions
    assert dimension_metadata_to_dict(bound.dimensions) == document


def test_bindable_from_meta_rejects_dimensions_for_undeclared_variables() -> None:
    meta = model_meta(PublicHookModel)
    dimensions = dimension_metadata_from_dict(
        {"dims": {"not_declared": ["obs"]}, "coords": {"obs": ["a", "b"]}}
    )

    with pytest.raises(ValueError, match="not declared by the model"):
        bindable_from_meta(meta, dimensions=dimensions)


def test_bind_model_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        bind_model(object, {})
