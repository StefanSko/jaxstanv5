from __future__ import annotations

import math

import pytest

from jaxstanv5 import Dim
from jaxstanv5.model.dimensions import (
    ResolvedModelDimensions,
    ResolvedVariableDims,
    dimension_metadata_to_dict,
    model_dimensions,
)


def test_dim_stores_name_and_optional_coords() -> None:
    assert Dim("predictor").name == "predictor"
    assert Dim("predictor").coords is None
    assert Dim("predictor", coords=("x1", "x2")).coords == ("x1", "x2")


def test_dim_accepts_json_scalar_coordinates() -> None:
    dim = Dim("mixed", coords=("x1", 1, 1.5, True, False, None))

    assert dim.coords == ("x1", 1, 1.5, True, False, None)


def test_dim_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Dim("")


def test_dim_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="strings"):
        Dim(1)  # type: ignore[arg-type]


def test_dim_rejects_non_json_scalar_coordinate() -> None:
    with pytest.raises(TypeError, match="JSON scalar"):
        Dim("predictor", coords=(object(),))


def test_dim_rejects_non_finite_float_coordinate() -> None:
    with pytest.raises(ValueError, match="finite"):
        Dim("predictor", coords=(math.nan,))

    with pytest.raises(ValueError, match="finite"):
        Dim("predictor", coords=(math.inf,))


def test_dimension_metadata_to_dict_returns_json_ready_lists() -> None:
    metadata = ResolvedModelDimensions(
        variables={
            "beta": ResolvedVariableDims(("predictor",)),
            "alpha": ResolvedVariableDims(()),
        },
        coords={"predictor": ("x1", "x2")},
    )

    assert dimension_metadata_to_dict(metadata) == {
        "dims": {"beta": ["predictor"], "alpha": []},
        "coords": {"predictor": ["x1", "x2"]},
    }


def test_model_dimensions_rejects_undecorated_classes() -> None:
    class Undecorated:
        pass

    with pytest.raises(TypeError, match="decorate it with @model"):
        model_dimensions(Undecorated)
