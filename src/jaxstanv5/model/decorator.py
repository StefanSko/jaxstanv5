"""Model declaration phase transitions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model._pending import (
    PendingBinOp,
    PendingConst,
    PendingDataRef,
    PendingExprNode,
    PendingIndexOp,
    PendingParamRef,
    UnresolvedSymbol,
    is_pending_expr,
    to_pending_expr,
)
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ExprNode, IndexOp, ParamRef

type ModelClass = type[object]


class FieldNormalizer(Protocol):
    def __call__(self, value: object) -> object: ...


@dataclass(frozen=True)
class PendingParam:
    """Parameter metadata during the pending declaration phase."""

    distribution: Distribution
    constraint: Constraint | None
    size: PendingDataRef | int | None
    symbol: UnresolvedSymbol


@dataclass(frozen=True)
class PendingObserved:
    """Observed metadata during the pending declaration phase."""

    distribution: Distribution


@dataclass(frozen=True)
class PendingModel:
    """Collected model metadata before symbols are resolved to names."""

    params: dict[str, PendingParam]
    data_slots: list[str]
    observed_name: str
    observed: PendingObserved
    expressions: dict[str, PendingExprNode]
    symbols: dict[UnresolvedSymbol, str]


@dataclass(frozen=True)
class ResolvedParam:
    """Parameter metadata after symbols are resolved to names."""

    distribution: Distribution
    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedObserved:
    """Observed metadata after symbols are resolved to names."""

    distribution: Distribution


@dataclass(frozen=True)
class ModelMeta:
    """Final model metadata attached by ``@model``."""

    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_name: str
    observed: ResolvedObserved
    expressions: dict[str, ExprNode]


def collect_pending_model(cls: ModelClass) -> PendingModel:
    """Collect class-body declarations into a pending model."""
    params: dict[str, PendingParam] = {}
    data_slots: list[str] = []
    symbols: dict[UnresolvedSymbol, str] = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param | Data):
            symbols[value.symbol] = name

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            params[name] = PendingParam(
                distribution=normalize_distribution_to_pending(value.distribution),
                constraint=value.constraint,
                size=normalize_size_to_pending(value.size),
                symbol=value.symbol,
            )
        elif isinstance(value, Data):
            data_slots.append(name)

    observed_name: str | None = None
    observed: PendingObserved | None = None
    expressions: dict[str, PendingExprNode] = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Observed):
            if observed_name is not None:
                raise ValueError("Model declarations must contain exactly one Observed")
            observed_name = name
            observed = PendingObserved(normalize_distribution_to_pending(value.distribution))
        elif is_pending_expr(value):
            expressions[name] = to_pending_expr(value)

    if observed_name is None or observed is None:
        raise ValueError("Model declarations must contain exactly one Observed")

    return PendingModel(
        params=params,
        data_slots=data_slots,
        observed_name=observed_name,
        observed=observed,
        expressions=expressions,
        symbols=symbols,
    )


def resolve_pending_model(pending: PendingModel) -> ModelMeta:
    """Resolve a pending model into final model metadata."""
    return ModelMeta(
        params={
            name: ResolvedParam(
                distribution=resolve_distribution(param.distribution, pending.symbols),
                constraint=param.constraint,
                size=resolve_size(param.size, pending.symbols),
            )
            for name, param in pending.params.items()
        },
        data_slots=pending.data_slots,
        observed_name=pending.observed_name,
        observed=ResolvedObserved(
            resolve_distribution(pending.observed.distribution, pending.symbols)
        ),
        expressions={
            name: resolve_expr(expr, pending.symbols) for name, expr in pending.expressions.items()
        },
    )


def model(cls: ModelClass) -> ModelClass:
    """Attach final model metadata to a declaration class."""
    pending = collect_pending_model(cls)
    meta = resolve_pending_model(pending)
    setattr(cls, "_model_meta", meta)  # noqa: B010
    setattr(cls, "bind", classmethod(make_bind(meta)))  # noqa: B010
    return cls


def make_bind(meta: ModelMeta) -> Callable[[ModelClass], object]:
    """Create a classmethod-compatible bind function for model metadata."""

    def bind(_cls: ModelClass, **values: object) -> object:
        from jaxstanv5.model.bound import BoundModel

        expected = set(meta.data_slots)
        expected.add(meta.observed_name)
        actual = set(values)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"Missing model data: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected model data: {sorted(extra)}")

        data = {name: jnp.asarray(value) for name, value in values.items()}
        param_shapes = {
            name: resolve_param_shape(param.size, data) for name, param in meta.params.items()
        }
        n_params = sum(param_count(shape) for shape in param_shapes.values())
        return BoundModel(meta=meta, data=data, param_shapes=param_shapes, n_params=n_params)

    return bind


def resolve_param_shape(
    size: DataRef | int | None,
    data: dict[str, jax.Array],
) -> tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, int):
        return (size,)
    return (int(data[size.name]),)


def param_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def normalize_size_to_pending(size: object) -> PendingDataRef | int | None:
    if size is None or isinstance(size, int):
        return size
    if isinstance(size, Data):
        return size.ref()
    if isinstance(size, PendingDataRef):
        return size
    raise TypeError(f"Cannot convert {type(size).__name__} to a pending size")


def normalize_distribution_to_pending(distribution: Distribution) -> Distribution:
    return rebuild_distribution(distribution, normalize_distribution_field_to_pending)


def normalize_distribution_field_to_pending(value: object) -> object:
    if isinstance(value, Param | Data) or is_pending_expr(value) or isinstance(value, int | float):
        return to_pending_expr(value)
    if is_dataclass(value) and not isinstance(value, type):
        return normalize_distribution_to_pending(cast(Distribution, value))
    return value


def resolve_size(
    size: PendingDataRef | int | None,
    symbols: dict[UnresolvedSymbol, str],
) -> DataRef | int | None:
    if size is None or isinstance(size, int):
        return size
    return DataRef(symbols[size.name])


def resolve_distribution(
    distribution: Distribution,
    symbols: dict[UnresolvedSymbol, str],
) -> Distribution:
    return rebuild_distribution(
        distribution, lambda value: resolve_distribution_field(value, symbols)
    )


def resolve_distribution_field(
    value: object,
    symbols: dict[UnresolvedSymbol, str],
) -> object:
    if is_pending_expr(value):
        return resolve_expr(to_pending_expr(value), symbols)
    if is_dataclass(value) and not isinstance(value, type):
        return resolve_distribution(cast(Distribution, value), symbols)
    return value


def resolve_expr(
    expr: PendingExprNode,
    symbols: dict[UnresolvedSymbol, str],
) -> ExprNode:
    if isinstance(expr, PendingParamRef):
        return ParamRef(symbols[expr.name])
    if isinstance(expr, PendingDataRef):
        return DataRef(symbols[expr.name])
    if isinstance(expr, PendingConst):
        return ConstNode(expr.value)
    if isinstance(expr, PendingBinOp):
        return BinOp(expr.op, resolve_expr(expr.left, symbols), resolve_expr(expr.right, symbols))
    if isinstance(expr, PendingIndexOp):
        return IndexOp(resolve_expr(expr.base, symbols), resolve_expr(expr.index, symbols))


def rebuild_distribution(
    distribution: Distribution,
    normalize_field: FieldNormalizer,
) -> Distribution:
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution
    normalized = {
        field.name: normalize_field(getattr(distribution, field.name))
        for field in fields(distribution)
    }
    return type(distribution)(**normalized)
