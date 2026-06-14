"""Shared resolved-metadata builders for IR serialization tests."""

from __future__ import annotations

from jaxstanv5.constraints import Interval, Ordered, Positive, UnitInterval
from jaxstanv5.distributions import MultivariateNormal, Normal
from jaxstanv5.model._data_schema import (
    DataDimRef,
    ResolvedDataRankSchema,
    ResolvedDataShapeSchema,
)
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedData,
    ResolvedFreeValue,
    ResolvedObserved,
    ResolvedParam,
    ResolvedStochasticSite,
)
from jaxstanv5.model.expr import (
    BinOp,
    DataRef,
    ExprNode,
    FullSlice,
    IndexOp,
    IndexTuple,
    ParamRef,
    ScalarIndex,
    UnaryOp,
    VectorScatterOp,
)


def minimal_meta(*, mu: ExprNode | None = None) -> ModelMeta:
    """Return a small regression-shaped metadata value with one expression."""
    expression = mu if mu is not None else BinOp("+", ParamRef("alpha"), DataRef("x"))
    return ModelMeta(
        params={"alpha": ResolvedParam(Normal(0.0, 1.0), constraint=None, size=None)},
        data={"x": ResolvedData(ResolvedDataRankSchema(1))},
        observed_nodes=(ResolvedObserved("y", Normal(expression, 1.0)),),
        expressions={"mu": expression},
        free_values={"alpha": ResolvedFreeValue(constraint=None, size=None)},
        stochastic_sites=(
            ResolvedStochasticSite("alpha", Normal(0.0, 1.0), ParamRef("alpha")),
            ResolvedStochasticSite("y", Normal(expression, 1.0), DataRef("y")),
        ),
    )


def rich_meta() -> ModelMeta:
    """Return metadata exercising every constraint, schema, and expression node."""
    indexed = IndexOp(
        DataRef("x"),
        IndexTuple((ScalarIndex(DataRef("idx")), FullSlice())),
    )
    scatter = VectorScatterOp(
        length=DataRef("n"),
        observed_idx=DataRef("obs_idx"),
        observed_values=DataRef("y_obs"),
        missing_idx=DataRef("mis_idx"),
        missing_values=ParamRef("w"),
    )
    return ModelMeta(
        params={
            "sigma": ResolvedParam(Normal(0.0, 1.0), constraint=Positive(), size=None),
            "cutpoints": ResolvedParam(Normal(0.0, 2.0), constraint=Ordered(), size=DataRef("n")),
            "p": ResolvedParam(Normal(0.0, 1.0), constraint=UnitInterval(), size=2),
            "level": ResolvedParam(Normal(0.0, 1.0), constraint=Interval(-1.0, 3.0), size=None),
        },
        data={
            "n": ResolvedData(ResolvedDataShapeSchema(())),
            "chol": ResolvedData(ResolvedDataShapeSchema((DataDimRef("n"), 4))),
            "x": ResolvedData(ResolvedDataRankSchema(2)),
        },
        observed_nodes=(ResolvedObserved("y", MultivariateNormal(0.0, DataRef("chol"))),),
        expressions={"eta": UnaryOp("exp", indexed)},
        free_values={
            "sigma": ResolvedFreeValue(constraint=Positive(), size=None),
            "w": ResolvedFreeValue(constraint=None, size=DataRef("n")),
        },
        stochastic_sites=(
            ResolvedStochasticSite("sigma", Normal(0.0, 1.0), ParamRef("sigma")),
            ResolvedStochasticSite("w", MultivariateNormal(0.0, DataRef("chol")), scatter),
        ),
    )
