from __future__ import annotations

import math

import pytest

from jaxstanv5 import Data, Dim, Observed, Param, model
from jaxstanv5.distributions import Normal
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


def test_data_declarations_accept_matching_dimension_rank() -> None:
    obs = Dim("obs")
    predictor = Dim("predictor")

    assert Data.scalar(dims=()).dims == ()
    assert Data.vector(3, dims=(obs,)).dims == (obs,)
    assert Data.matrix(3, 2, dims=(obs, predictor)).dims == (obs, predictor)
    assert Data.array(shape=(3, 2), dims=(obs, predictor)).dims == (obs, predictor)
    assert Data.array(rank=2, dims=(obs, predictor)).dims == (obs, predictor)


def test_data_declarations_reject_dimension_rank_mismatch() -> None:
    obs = Dim("obs")
    predictor = Dim("predictor")

    with pytest.raises(ValueError, match="Data dims length"):
        Data.scalar(dims=(obs,))
    with pytest.raises(ValueError, match="Data dims length"):
        Data.vector(3, dims=(obs, predictor))
    with pytest.raises(ValueError, match="Data dims length"):
        Data.matrix(3, 2, dims=(obs,))


def test_param_declarations_accept_matching_dimension_rank() -> None:
    predictor = Dim("predictor")

    assert Param(Normal(0.0, 1.0), dims=()).dims == ()
    assert Param(Normal(0.0, 1.0), size=3, dims=(predictor,)).dims == (predictor,)


def test_param_declarations_reject_dimension_rank_mismatch() -> None:
    predictor = Dim("predictor")

    with pytest.raises(ValueError, match="Param dims length"):
        Param(Normal(0.0, 1.0), dims=(predictor,))
    with pytest.raises(ValueError, match="Param dims length"):
        Param(Normal(0.0, 1.0), size=3, dims=())


def test_observed_declarations_store_dimension_labels() -> None:
    obs = Dim("obs")

    assert Observed(Normal(0.0, 1.0), dims=(obs,)).dims == (obs,)
    assert Observed(Normal(0.0, 1.0), dims=()).dims == ()


def test_model_dimension_resolution_accepts_duplicate_matching_coords() -> None:
    predictor_for_data = Dim("predictor", coords=("x1", "x2"))
    predictor_for_param = Dim("predictor", coords=("x1", "x2"))
    obs = Dim("obs")

    @model
    class MatchingCoords:
        x = Data.matrix(3, 2, dims=(obs, predictor_for_data))
        beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor_for_param,))
        y = Observed(Normal(0.0, 1.0), dims=(obs,))

    assert model_dimensions(MatchingCoords).coords == {"predictor": ("x1", "x2")}


def test_model_dimension_resolution_rejects_conflicting_coords() -> None:
    predictor_for_data = Dim("predictor", coords=("x1", "x2"))
    predictor_for_param = Dim("predictor", coords=("a", "b"))
    obs = Dim("obs")

    with pytest.raises(ValueError, match="conflicting coordinate values"):

        @model
        class ConflictingCoords:
            x = Data.matrix(3, 2, dims=(obs, predictor_for_data))
            beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor_for_param,))
            y = Observed(Normal(0.0, 1.0), dims=(obs,))


def test_model_dimension_resolution_rejects_static_coord_size_mismatch() -> None:
    predictor = Dim("predictor", coords=("x1", "x2"))
    obs = Dim("obs")

    with pytest.raises(ValueError, match="coordinate length"):

        @model
        class DataCoordSizeMismatch:
            x = Data.matrix(3, 3, dims=(obs, predictor))
            beta = Param(Normal(0.0, 1.0))
            y = Observed(Normal(0.0, 1.0), dims=(obs,))

    with pytest.raises(ValueError, match="coordinate length"):

        @model
        class ParamCoordSizeMismatch:
            beta = Param(Normal(0.0, 1.0), size=3, dims=(predictor,))
