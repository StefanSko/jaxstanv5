"""Model declaration resolution and binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    is_deferred_expr,
)
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ExprNode, IndexOp, ParamRef

type ModelClass = type[object]
type SymbolTable = dict[DeclarationSymbol, str]


@dataclass(frozen=True)
class ResolvedParam:
    """Parameter metadata after declaration symbols are resolved to names."""

    distribution: Distribution
    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedObserved:
    """Observed metadata after declaration symbols are resolved to names."""

    distribution: Distribution


@dataclass(frozen=True)
class ModelMeta:
    """Final model metadata attached by ``@model``."""

    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_name: str
    observed: ResolvedObserved
    expressions: dict[str, ExprNode]


@dataclass(frozen=True)
class _ResolvedDeclarations:
    """Resolved top-level declarations from a declaration class."""

    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_name: str
    observed: ResolvedObserved


def _resolve_model_declaration(cls: ModelClass) -> ModelMeta:
    """Resolve a declaration class into final model metadata."""
    symbols = _collect_declaration_symbols(cls)
    declarations = _resolve_declarations(cls, symbols)
    expressions = _resolve_expressions(cls, symbols)

    return ModelMeta(
        params=declarations.params,
        data_slots=declarations.data_slots,
        observed_name=declarations.observed_name,
        observed=declarations.observed,
        expressions=expressions,
    )


def _collect_declaration_symbols(cls: ModelClass) -> SymbolTable:
    """Collect declaration symbols and reject declaration aliases."""
    symbols: SymbolTable = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param | Data):
            existing_name = symbols.get(value.symbol)
            if existing_name is not None:
                raise ValueError(
                    "Declaration aliases are not supported: "
                    f"{existing_name!r} and {name!r} share one symbol"
                )
            symbols[value.symbol] = name

    return symbols


def _resolve_declarations(cls: ModelClass, symbols: SymbolTable) -> _ResolvedDeclarations:
    """Resolve top-level declaration inventory into final named metadata."""
    params: dict[str, ResolvedParam] = {}
    data_slots: list[str] = []
    observed_name: str | None = None
    observed: ResolvedObserved | None = None

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            params[name] = ResolvedParam(
                distribution=_resolve_declaration_distribution(value.distribution, symbols),
                constraint=value.constraint,
                size=_resolve_declaration_size(value.size, symbols),
            )
        elif isinstance(value, Data):
            data_slots.append(name)
        elif isinstance(value, Observed):
            if observed_name is not None:
                raise ValueError("Model declarations must contain exactly one Observed")
            observed_name = name
            observed = ResolvedObserved(
                _resolve_declaration_distribution(value.distribution, symbols)
            )

    if observed_name is None or observed is None:
        raise ValueError("Model declarations must contain exactly one Observed")

    return _ResolvedDeclarations(
        params=params,
        data_slots=data_slots,
        observed_name=observed_name,
        observed=observed,
    )


def _resolve_expressions(cls: ModelClass, symbols: SymbolTable) -> dict[str, ExprNode]:
    """Resolve top-level derived declaration expressions into final IR."""
    expressions: dict[str, ExprNode] = {}

    for name, value in cls.__dict__.items():
        if is_deferred_expr(value):
            expressions[name] = _resolve_declaration_expr(value, symbols)

    return expressions


def model(cls: ModelClass) -> ModelClass:
    """Attach final model metadata to a declaration class."""
    meta = _resolve_model_declaration(cls)
    setattr(cls, "_model_meta", meta)  # noqa: B010
    setattr(cls, "bind", classmethod(_make_bind(meta)))  # noqa: B010
    return cls


def _make_bind(meta: ModelMeta) -> Callable[[ModelClass], object]:
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
            name: _resolve_param_shape(param.size, data) for name, param in meta.params.items()
        }
        n_params = sum(_param_count(shape) for shape in param_shapes.values())
        return BoundModel(meta=meta, data=data, param_shapes=param_shapes, n_params=n_params)

    return bind


def _resolve_param_shape(
    size: DataRef | int | None,
    data: dict[str, jax.Array],
) -> tuple[int, ...]:
    if size is None:
        return ()
    if isinstance(size, int):
        return (_validate_parameter_size(size, "Parameter size"),)

    size_value = data[size.name]
    if size_value.ndim != 0:
        raise ValueError(f"Data-dependent parameter size {size.name!r} must be scalar")
    if not jnp.issubdtype(size_value.dtype, jnp.integer):
        raise TypeError(f"Data-dependent parameter size {size.name!r} must be integer")
    return (
        _validate_parameter_size(
            int(size_value),
            f"Data-dependent parameter size {size.name!r}",
        ),
    )


def _param_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        if dim < 0:
            raise ValueError("Parameter shape dimensions must be non-negative")
        count *= dim
    return count


def _validate_parameter_size(size: int, label: str) -> int:
    if isinstance(size, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    if size < 0:
        raise ValueError(f"{label} must be non-negative")
    return size


def _resolve_declaration_size(size: object, symbols: SymbolTable) -> DataRef | int | None:
    """Resolve a declaration-size value into final size metadata."""
    if size is None:
        return None
    if isinstance(size, int):
        return _validate_parameter_size(size, "Parameter size")
    if isinstance(size, Data):
        return DataRef(_resolve_symbol(size.symbol, symbols))
    raise TypeError(f"Cannot resolve {type(size).__name__} as a declaration size")


def _resolve_declaration_distribution(
    distribution: Distribution,
    symbols: SymbolTable,
) -> Distribution:
    """Resolve symbolic distribution fields into final expression nodes."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution
    resolved = {
        field.name: _resolve_declaration_distribution_field(
            getattr(distribution, field.name),
            symbols,
        )
        for field in fields(distribution)
    }
    return type(distribution)(**resolved)


def _resolve_declaration_distribution_field(value: object, symbols: SymbolTable) -> object:
    if _is_declaration_expr(value):
        return _resolve_declaration_expr(value, symbols)
    if _is_final_expr_node(value):
        raise TypeError("Final expression nodes are not valid in model declarations")
    if is_dataclass(value) and not isinstance(value, type):
        return _resolve_declaration_distribution(cast(Distribution, value), symbols)
    return value


def _resolve_declaration_expr(value: object, symbols: SymbolTable) -> ExprNode:
    """Resolve class-body declaration syntax into final expression IR."""
    if isinstance(value, Param):
        return ParamRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, Data):
        return DataRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, int | float):
        return ConstNode(value)
    if isinstance(value, DeferredBinOp):
        return BinOp(
            value.op,
            _resolve_declaration_expr(value.left, symbols),
            _resolve_declaration_expr(value.right, symbols),
        )
    if isinstance(value, DeferredIndexOp):
        return IndexOp(
            _resolve_declaration_expr(value.base, symbols),
            _resolve_declaration_expr(value.index, symbols),
        )
    raise TypeError(f"Cannot resolve {type(value).__name__} as a declaration expression")


def _is_declaration_expr(value: object) -> bool:
    """Return whether ``value`` can resolve to final expression IR."""
    return isinstance(value, Param | Data | DeferredBinOp | DeferredIndexOp | int | float)


def _is_final_expr_node(value: object) -> bool:
    """Return whether ``value`` is already resolved final expression IR."""
    return isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp)


def _resolve_symbol(symbol: DeclarationSymbol, symbols: SymbolTable) -> str:
    name = symbols.get(symbol)
    if name is None:
        raise ValueError(f"Unknown declaration symbol: {symbol}")
    return name
