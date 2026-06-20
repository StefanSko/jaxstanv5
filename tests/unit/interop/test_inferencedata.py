from __future__ import annotations

from typing import Protocol, cast

import jax.numpy as jnp

from jaxstanv5 import Data, Dim, Observed, Param, model, model_dimensions
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import NutsDiagnosticTrace, SamplerDiagnostics, SamplerResult
from jaxstanv5.interop.inferencedata import inferencedata_groups
from jaxstanv5.model.bound import BoundModel


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


def _bind(model_cls: object, **values: object) -> BoundModel:
    return cast(_BindableModel, model_cls).bind(**values)


def _diagnostics(*, chains: int, draws: int) -> SamplerDiagnostics:
    shape = (chains, draws)
    trace = NutsDiagnosticTrace(
        is_divergent=jnp.asarray([[False, True, False], [False, False, False]])[:chains, :draws],
        acceptance_rate=jnp.full(shape, 0.8),
        num_integration_steps=jnp.full(shape, 7),
        num_trajectory_expansions=jnp.full(shape, 3),
        energy=jnp.arange(chains * draws, dtype=jnp.float32).reshape(shape),
    )
    return SamplerDiagnostics(warmup=trace, sampling=trace)


def test_inferencedata_groups_use_declared_dims_coords_and_standard_sample_stats() -> None:
    predictor = Dim("predictor", coords=("x1", "x2", "x3"))
    obs = Dim("obs", coords=("row0", "row1"))

    @model
    class LinearRegression:
        x = Data.matrix(2, 3, dims=(obs, predictor))
        beta = Param(Normal(0.0, 1.0), size=3, dims=(predictor,))
        alpha = Param(Normal(0.0, 1.0))
        y = Observed(Normal(alpha, 1.0), dims=(obs,))

    bound = _bind(
        LinearRegression,
        x=jnp.ones((2, 3)),
        y=jnp.asarray([0.5, -0.5]),
    )
    result = SamplerResult(
        samples={
            "beta": jnp.arange(18, dtype=jnp.float32).reshape(2, 3, 3),
            "alpha": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        },
        diagnostics=_diagnostics(chains=2, draws=3),
    )

    groups = inferencedata_groups(bound, result)

    assert bound.dimensions == model_dimensions(LinearRegression)
    assert groups.coords["chain"] == (0, 1)
    assert groups.coords["draw"] == (0, 1, 2)
    assert groups.coords["predictor"] == ("x1", "x2", "x3")
    assert groups.coords["obs"] == ("row0", "row1")

    assert groups.posterior.variables["beta"].dims == ("chain", "draw", "predictor")
    assert groups.posterior.variables["alpha"].dims == ("chain", "draw")
    assert groups.observed_data.variables["y"].dims == ("obs",)
    assert groups.constant_data.variables["x"].dims == ("obs", "predictor")

    assert groups.sample_stats.variables["diverging"].dims == ("chain", "draw")
    assert groups.sample_stats.variables["acceptance_rate"].dims == ("chain", "draw")
    assert groups.sample_stats.variables["n_steps"].dims == ("chain", "draw")
    assert groups.sample_stats.variables["tree_depth"].dims == ("chain", "draw")
    assert groups.sample_stats.variables["energy"].dims == ("chain", "draw")
    assert bool(groups.sample_stats.variables["diverging"].values[0, 1])


def test_inferencedata_groups_generate_stable_fallback_dims_and_coords() -> None:
    @model
    class VectorModel:
        theta = Param(Normal(0.0, 1.0), size=2)
        y = Observed(Normal(0.0, 1.0))

    bound = _bind(VectorModel, y=1.5)
    result = SamplerResult(
        samples={"theta": jnp.ones((1, 2, 2))},
        diagnostics=_diagnostics(chains=1, draws=2),
    )

    groups = inferencedata_groups(bound, result)

    assert groups.posterior.variables["theta"].dims == ("chain", "draw", "theta_dim_0")
    assert groups.coords["theta_dim_0"] == (0, 1)
    assert groups.observed_data.variables["y"].dims == ()
