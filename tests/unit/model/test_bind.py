"""Unit tests for binding helpers and parameter shape resolution."""

from __future__ import annotations

from typing import Protocol, cast

import jax
import jax.numpy as jnp
import pytest

from jaxstanv5._backends.jax.binding import _param_count, _resolve_param_shape
from jaxstanv5.distributions import Binomial, MultivariateNormal, Normal, OrderedLogistic
from jaxstanv5.model._data_schema import DataDimRef, ResolvedDataRankSchema, ResolvedDataShapeSchema
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedData,
    ResolvedFreeValue,
    ResolvedObserved,
    ResolvedParam,
    ResolvedStochasticSite,
    _make_bind,
)
from jaxstanv5.model.expr import DataRef, ParamRef, VectorScatterOp


class BindFn(Protocol):
    def __call__(self, _cls: type[object], **values: object) -> object: ...


def make_meta() -> ModelMeta:
    return ModelMeta(
        params={
            "alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None),
            "beta": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=2),
            "group": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=DataRef("n")),
        },
        data={
            "x": ResolvedData(ResolvedDataRankSchema(1)),
            "n": ResolvedData(ResolvedDataShapeSchema(())),
        },
        observed_nodes=(ResolvedObserved("y", Normal(0.0, 1.0)),),
        expressions={},
    )


def bind_meta(meta: ModelMeta, **values: object) -> BoundModel:
    bind = cast(BindFn, _make_bind(meta))
    return cast(BoundModel, bind(type("Model", (), {}), **values))


def test_bind_requires_all_observed_node_values() -> None:
    meta = ModelMeta(
        params={},
        data={"x": ResolvedData(ResolvedDataShapeSchema(()))},
        observed_nodes=(
            ResolvedObserved("x_obs", Normal(0.0, 1.0)),
            ResolvedObserved("y_obs", Normal(0.0, 1.0)),
        ),
        expressions={},
    )

    with pytest.raises(ValueError, match=r"Missing model data: \['y_obs'\]"):
        bind_meta(meta, x=0.0, x_obs=1.0)


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


def test_bind_rejects_nan_bound_values() -> None:
    with pytest.raises(ValueError, match="contains NaN"):
        bind_meta(make_meta(), x=jnp.asarray([jnp.nan]), n=1, y=jnp.asarray([0.0]))


def test_bind_rejects_infinite_bound_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        bind_meta(make_meta(), x=jnp.asarray([1.0]), n=1, y=jnp.asarray([jnp.inf]))


def test_bind_rejects_wrong_data_rank() -> None:
    with pytest.raises(ValueError, match="Data 'x' has wrong rank"):
        bind_meta(make_meta(), x=jnp.asarray(1.0), n=3, y=jnp.asarray([0.0, 1.0]))


def test_bind_rejects_wrong_data_shape() -> None:
    meta = shaped_chol_meta()

    with pytest.raises(ValueError, match="Data 'chol' has wrong shape"):
        bind_meta(meta, n=3, chol=jnp.eye(2), y=0.0)


def test_bind_rejects_non_scalar_dynamic_data_shape_dimension_source() -> None:
    with pytest.raises(ValueError, match="Data 'n' has wrong shape"):
        bind_meta(shaped_chol_meta(), n=jnp.asarray([3]), chol=jnp.eye(3), y=0.0)


def test_bind_rejects_non_integer_dynamic_data_shape_dimension() -> None:
    with pytest.raises(TypeError, match="Data shape dimension 'n' must be integer"):
        bind_meta(shaped_chol_meta(), n=3.0, chol=jnp.eye(3), y=0.0)


def shaped_chol_meta() -> ModelMeta:
    return ModelMeta(
        params={},
        data={
            "n": ResolvedData(ResolvedDataShapeSchema(())),
            "chol": ResolvedData(ResolvedDataShapeSchema((DataDimRef("n"), DataDimRef("n")))),
        },
        observed_nodes=(ResolvedObserved("y", Normal(0.0, 1.0)),),
        expressions={},
    )


def test_bind_rejects_overlapping_partial_vector_indexes() -> None:
    with pytest.raises(ValueError, match="Partial-observed indexes must be disjoint"):
        bind_meta(
            partial_vector_meta(),
            n=3,
            n_mis=1,
            obs_idx=jnp.asarray([0, 1]),
            mis_idx=jnp.asarray([1]),
            y_obs=jnp.asarray([1.0, 2.0]),
        )


def partial_vector_meta() -> ModelMeta:
    return ModelMeta(
        params={},
        data={
            "n": ResolvedData(ResolvedDataShapeSchema(())),
            "n_mis": ResolvedData(ResolvedDataShapeSchema(())),
            "obs_idx": ResolvedData(ResolvedDataRankSchema(1)),
            "mis_idx": ResolvedData(ResolvedDataRankSchema(1)),
            "y_obs": ResolvedData(ResolvedDataRankSchema(1)),
        },
        observed_nodes=(),
        expressions={},
        free_values={"y": ResolvedFreeValue(constraint=None, size=DataRef("n_mis"))},
        stochastic_sites=(
            ResolvedStochasticSite(
                name="y",
                distribution=Normal(0.0, 1.0),
                value=VectorScatterOp(
                    length=DataRef("n"),
                    observed_idx=DataRef("obs_idx"),
                    observed_values=DataRef("y_obs"),
                    missing_idx=DataRef("mis_idx"),
                    missing_values=ParamRef("y"),
                ),
            ),
        ),
    )


def test_bind_rejects_observed_values_that_expand_against_distribution_shape() -> None:
    meta = ModelMeta(
        params={"alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None)},
        data={"x": ResolvedData(ResolvedDataRankSchema(1))},
        observed_nodes=(ResolvedObserved("y", Normal(DataRef("x"), 1.0)),),
        expressions={},
        stochastic_sites=(
            ResolvedStochasticSite("alpha", Normal(0.0, 1.0), ParamRef("alpha")),
            ResolvedStochasticSite("y", Normal(DataRef("x"), 1.0), DataRef("y")),
        ),
    )

    with pytest.raises(ValueError, match="would broadcast"):
        bind_meta(meta, x=jnp.arange(5.0), y=jnp.ones((5, 1)))


def test_bind_rejects_scalar_parameter_with_vector_batch_prior() -> None:
    meta = ModelMeta(
        params={"beta": ResolvedParam(Normal(DataRef("x"), 1.0), constraint=None, size=None)},
        data={"x": ResolvedData(ResolvedDataRankSchema(1))},
        observed_nodes=(),
        expressions={},
        free_values={"beta": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(
            ResolvedStochasticSite("beta", Normal(DataRef("x"), 1.0), ParamRef("beta")),
        ),
    )

    with pytest.raises(ValueError, match="would broadcast"):
        bind_meta(meta, x=jnp.arange(3.0))


def test_bind_rejects_multivariate_parameter_without_event_size() -> None:
    meta = ModelMeta(
        params={
            "theta": ResolvedParam(
                MultivariateNormal(jnp.zeros((2,)), jnp.eye(2)),
                constraint=None,
                size=None,
            )
        },
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"theta": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(
            ResolvedStochasticSite(
                "theta",
                MultivariateNormal(jnp.zeros((2,)), jnp.eye(2)),
                ParamRef("theta"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="would broadcast"):
        bind_meta(meta)


def test_bind_accepts_vector_parameter_with_scalar_iid_prior() -> None:
    meta = ModelMeta(
        params={"beta": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=3)},
        data={},
        observed_nodes=(),
        expressions={},
        free_values={"beta": ResolvedFreeValue(constraint=None, size=3)},
        stochastic_sites=(ResolvedStochasticSite("beta", Normal(0.0, 1.0), ParamRef("beta")),),
    )

    bound = bind_meta(meta)

    assert bound.param_shapes == {"beta": (3,)}


def test_bind_rejects_non_integer_discrete_observation() -> None:
    dist = Binomial(5.0, 0.5)
    meta = ModelMeta(
        params={},
        data={},
        observed_nodes=(ResolvedObserved("y", dist),),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("y", dist, DataRef("y")),),
    )

    with pytest.raises(ValueError, match="non-integer"):
        bind_meta(meta, y=jnp.asarray(2.5))


def test_bind_rejects_binomial_observation_above_concrete_total_count() -> None:
    dist = Binomial(DataRef("n"), 0.5)
    meta = ModelMeta(
        params={},
        data={"n": ResolvedData(ResolvedDataShapeSchema(()))},
        observed_nodes=(ResolvedObserved("y", dist),),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("y", dist, DataRef("y")),),
    )

    with pytest.raises(ValueError, match="total_count"):
        bind_meta(meta, n=5, y=jnp.asarray(6.0))


def test_bind_rejects_ordered_logistic_observation_outside_category_range() -> None:
    dist = OrderedLogistic(0.0, DataRef("cutpoints"))
    meta = ModelMeta(
        params={},
        data={"cutpoints": ResolvedData(ResolvedDataRankSchema(1))},
        observed_nodes=(ResolvedObserved("y", dist),),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("y", dist, DataRef("y")),),
    )

    with pytest.raises(ValueError, match="between"):
        bind_meta(meta, cutpoints=jnp.asarray([-1.0, 1.0]), y=jnp.asarray(3.0))


def test_bind_rejects_non_lower_triangular_mvn_scale_tril_data() -> None:
    dist = MultivariateNormal(0.0, DataRef("chol"))
    meta = ModelMeta(
        params={},
        data={"chol": ResolvedData(ResolvedDataRankSchema(2))},
        observed_nodes=(ResolvedObserved("y", dist),),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("y", dist, DataRef("y")),),
    )

    with pytest.raises(ValueError, match="jnp.linalg.cholesky"):
        bind_meta(
            meta,
            chol=jnp.asarray([[1.0, 0.5], [0.0, 1.0]]),
            y=jnp.zeros((2,)),
        )


def test_bind_rejects_non_positive_mvn_scale_tril_diagonal_data() -> None:
    dist = MultivariateNormal(0.0, DataRef("chol"))
    meta = ModelMeta(
        params={},
        data={"chol": ResolvedData(ResolvedDataRankSchema(2))},
        observed_nodes=(ResolvedObserved("y", dist),),
        expressions={},
        stochastic_sites=(ResolvedStochasticSite("y", dist, DataRef("y")),),
    )

    with pytest.raises(ValueError, match="strictly positive"):
        bind_meta(
            meta,
            chol=jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
            y=jnp.zeros((2,)),
        )


def test_resolve_param_shape_rejects_negative_literal_size() -> None:
    with pytest.raises(ValueError, match="Parameter size must be non-negative"):
        _resolve_param_shape(-1, {})


def test_resolve_param_shape_rejects_bool_literal_size() -> None:
    with pytest.raises(TypeError, match="Parameter size must be an integer, not bool"):
        _resolve_param_shape(True, {})


def test_resolve_param_shape_rejects_non_scalar_data_size() -> None:
    with pytest.raises(ValueError, match="Data-dependent parameter size 'n' must be scalar"):
        _resolve_param_shape(DataRef("n"), {"n": jnp.asarray([2, 3])})


def test_resolve_param_shape_rejects_float_data_size() -> None:
    with pytest.raises(TypeError, match="Data-dependent parameter size 'n' must be integer"):
        _resolve_param_shape(DataRef("n"), {"n": jnp.asarray(2.5)})


def test_resolve_param_shape_rejects_negative_data_size() -> None:
    with pytest.raises(ValueError, match="Data-dependent parameter size 'n' must be non-negative"):
        _resolve_param_shape(DataRef("n"), {"n": jnp.asarray(-1)})


def test_param_count_multiplies_dimensions_and_rejects_negative_dimensions() -> None:
    assert _param_count(()) == 1
    assert _param_count((2, 3, 4)) == 24

    with pytest.raises(ValueError, match="Parameter shape dimensions must be non-negative"):
        _param_count((2, -1))
