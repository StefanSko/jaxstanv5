"""bind_model over bayeswire's public hooks: the authoring-to-backend transition."""

from __future__ import annotations

from typing import Protocol, cast

import pytest
from bayeswire.distributions import Normal
from bayeswire.ir import bindable_from_meta
from bayeswire.model import (
    Dim,
    Observed,
    Param,
    dimension_metadata_from_dict,
    dimension_metadata_to_dict,
    model,
    model_dimensions,
    model_meta,
)

from jaxstanv5.model import BoundModel, bind_model


class _Shaped(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...


obs = Dim("obs", coords=("a", "b"))


@model
class PublicHookModel:
    theta = Param(Normal(0.0, 1.0))
    y = Observed(Normal(theta, 1.0), dims=(obs,))


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


def test_bind_model_accepts_ir_bindable_models() -> None:
    meta = model_meta(PublicHookModel)
    rebuilt = bindable_from_meta(meta)

    bound = bind_model(rebuilt, {"y": [1.0, 2.0]})
    assert bound.meta is meta
    # Without the decoded dimension sidecar there is nothing to reattach.
    assert bound.dimensions is None


def test_bind_model_reattaches_round_tripped_dimension_metadata() -> None:
    meta = model_meta(PublicHookModel)
    declared_dimensions = model_dimensions(PublicHookModel)
    document = dimension_metadata_to_dict(declared_dimensions)
    rebuilt = bindable_from_meta(meta, dimensions=dimension_metadata_from_dict(document))

    bound = bind_model(rebuilt, {"y": [1.0, 2.0]})
    assert bound.dimensions is not None
    assert bound.dimensions == declared_dimensions
    assert dimension_metadata_to_dict(bound.dimensions) == document


def test_bind_model_rejects_non_model_objects() -> None:
    with pytest.raises(TypeError, match="@model"):
        bind_model(object, {})


def test_bind_model_rejects_subclasses_with_inherited_model_metadata() -> None:
    class ChildModel(PublicHookModel):
        extra = Param(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="@model"):
        bind_model(ChildModel, {"y": [1.0, 2.0]})


def test_bind_model_rejects_non_mapping_values() -> None:
    with pytest.raises(TypeError, match="mapping"):
        bind_model(PublicHookModel, cast("dict[str, object]", [("y", [1.0, 2.0])]))
