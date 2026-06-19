from __future__ import annotations

from jaxstanv5 import (
    Data,
    Dim,
    Observed,
    Param,
    dimension_metadata_to_dict,
    model,
    model_dimensions,
)
from jaxstanv5.distributions import Normal


def test_declared_dimension_metadata_is_exposed_for_linear_regression() -> None:
    predictor = Dim("predictor", coords=("x1", "x2", "x3"))
    obs = Dim("obs")

    @model
    class LinearRegression:
        x = Data.matrix(5, 3, dims=(obs, predictor))
        beta = Param(Normal(0.0, 1.0), size=3, dims=(predictor,))
        alpha = Param(Normal(0.0, 1.0))
        y = Observed(Normal(alpha, 1.0), dims=(obs,))

    assert dimension_metadata_to_dict(model_dimensions(LinearRegression)) == {
        "dims": {
            "x": ["obs", "predictor"],
            "beta": ["predictor"],
            "alpha": [],
            "y": ["obs"],
        },
        "coords": {"predictor": ["x1", "x2", "x3"]},
    }
