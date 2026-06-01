"""Model declaration resolution and binding."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import DiscreteDistribution, Distribution
from jaxstanv5.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    DeferredUnaryOp,
    is_deferred_expr,
)
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ExprNode, IndexOp, ParamRef, UnaryOp

type ModelClass = type[object]
type SymbolTable = dict[DeclarationSymbol, str]

_INDEX_BINOPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "+": jnp.add,
    "-": jnp.subtract,
    "*": jnp.multiply,
    "/": jnp.divide,
}

_INDEX_UNARY_FUNCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "exp": jnp.exp,
    "sigmoid": jax.nn.sigmoid,
}


@dataclass(frozen=True)
class ResolvedParam:
    """Parameter metadata after declaration symbols are resolved to names."""

    distribution: Distribution
    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedObserved:
    """Observed likelihood metadata after declaration symbols are resolved to names."""

    name: str
    distribution: Distribution


@dataclass(frozen=True)
class ModelMeta:
    """Final model metadata attached by ``@model``."""

    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_nodes: tuple[ResolvedObserved, ...]
    expressions: dict[str, ExprNode]


@dataclass(frozen=True)
class _ResolvedDeclarations:
    """Resolved top-level declarations from a declaration class."""

    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_nodes: tuple[ResolvedObserved, ...]


def _resolve_model_declaration(cls: ModelClass) -> ModelMeta:
    """Resolve a declaration class into final model metadata."""
    symbols = _collect_declaration_symbols(cls)
    declarations = _resolve_declarations(cls, symbols)
    expressions = _resolve_expressions(cls, symbols)

    return ModelMeta(
        params=declarations.params,
        data_slots=declarations.data_slots,
        observed_nodes=declarations.observed_nodes,
        expressions=expressions,
    )


def _collect_declaration_symbols(cls: ModelClass) -> SymbolTable:
    """Collect declaration symbols and reject declaration aliases."""
    symbols: SymbolTable = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param | Data | Observed):
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
    observed_nodes: list[ResolvedObserved] = []

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            if isinstance(distribution, DiscreteDistribution):
                raise TypeError(
                    "Discrete distributions cannot be used as Param priors; "
                    "use them for Observed likelihoods or marginalize discrete latents"
                )
            params[name] = ResolvedParam(
                distribution=distribution,
                constraint=value.constraint,
                size=_resolve_declaration_size(value.size, symbols),
            )
        elif isinstance(value, Data):
            data_slots.append(name)
        elif isinstance(value, Observed):
            observed_nodes.append(
                ResolvedObserved(
                    name=name,
                    distribution=_resolve_declaration_distribution(value.distribution, symbols),
                )
            )

    if not params and not observed_nodes:
        raise ValueError("Model declarations must contain at least one stochastic declaration")

    return _ResolvedDeclarations(
        params=params,
        data_slots=data_slots,
        observed_nodes=tuple(observed_nodes),
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
        expected.update(observed.name for observed in meta.observed_nodes)
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
        _validate_bound_index_expressions(meta, data, param_shapes)
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


def _validate_bound_index_expressions(
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate concrete data indexes before JAX gather semantics can clamp them."""
    for param in meta.params.values():
        _validate_distribution_index_expressions(param.distribution, data, param_shapes)
    for observed in meta.observed_nodes:
        _validate_distribution_index_expressions(observed.distribution, data, param_shapes)
    for expression in meta.expressions.values():
        _validate_index_expr(expression, data, param_shapes)


def _validate_distribution_index_expressions(
    distribution: Distribution,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate indexes inside symbolic dataclass distribution fields."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return

    for field in fields(distribution):
        value = getattr(distribution, field.name)
        if _is_final_expr_node(value):
            _validate_index_expr(cast(ExprNode, value), data, param_shapes)
        elif is_dataclass(value) and not isinstance(value, type):
            _validate_distribution_index_expressions(cast(Distribution, value), data, param_shapes)


def _validate_index_expr(
    node: ExprNode,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate all concrete indexes in a final expression tree."""
    if isinstance(node, BinOp):
        _validate_index_expr(node.left, data, param_shapes)
        _validate_index_expr(node.right, data, param_shapes)
    elif isinstance(node, UnaryOp):
        _validate_index_expr(node.operand, data, param_shapes)
    elif isinstance(node, IndexOp):
        _validate_index_expr(node.base, data, param_shapes)
        _validate_index_expr(node.index, data, param_shapes)
        _validate_single_index_op(node, data, param_shapes)


def _validate_single_index_op(
    node: IndexOp,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate one concrete first-axis index operation."""
    base_shape = _infer_expr_shape(node.base, data, param_shapes)
    if not base_shape:
        raise ValueError("Cannot index scalar expression")

    index = _evaluate_data_index_expr(node.index, data)
    _validate_index_value(index, base_shape[0])


def _infer_expr_shape(
    node: ExprNode,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    """Infer the concrete shape of a resolved expression at bind time."""
    if isinstance(node, ParamRef):
        return param_shapes[node.name]
    if isinstance(node, DataRef):
        return data[node.name].shape
    if isinstance(node, ConstNode):
        return jnp.asarray(node.value).shape
    if isinstance(node, BinOp):
        left_shape = _infer_expr_shape(node.left, data, param_shapes)
        right_shape = _infer_expr_shape(node.right, data, param_shapes)
        return jnp.broadcast_shapes(left_shape, right_shape)
    if isinstance(node, UnaryOp):
        return _infer_expr_shape(node.operand, data, param_shapes)
    if isinstance(node, IndexOp):
        base_shape = _infer_expr_shape(node.base, data, param_shapes)
        if not base_shape:
            raise ValueError("Cannot index scalar expression")
        index = _evaluate_data_index_expr(node.index, data)
        _validate_index_value(index, base_shape[0])
        return index.shape + base_shape[1:]
    raise TypeError(f"Cannot infer shape for expression node: {type(node).__name__}")


def _evaluate_data_index_expr(node: ExprNode, data: dict[str, jax.Array]) -> jax.Array:
    """Evaluate an index expression that must depend only on data and constants."""
    if isinstance(node, DataRef):
        return data[node.name]
    if isinstance(node, ConstNode):
        return jnp.asarray(node.value)
    if isinstance(node, ParamRef):
        raise TypeError("Index expressions must depend only on data or constants")
    if isinstance(node, BinOp):
        left = _evaluate_data_index_expr(node.left, data)
        right = _evaluate_data_index_expr(node.right, data)
        op_fn = _INDEX_BINOPS.get(node.op)
        if op_fn is None:
            raise ValueError(f"Unknown binary operator in index expression: {node.op!r}")
        return op_fn(left, right)
    if isinstance(node, UnaryOp):
        operand = _evaluate_data_index_expr(node.operand, data)
        function = _INDEX_UNARY_FUNCTIONS.get(node.function)
        if function is None:
            raise ValueError(f"Unknown unary function in index expression: {node.function!r}")
        return function(operand)
    if isinstance(node, IndexOp):
        base = _evaluate_data_index_expr(node.base, data)
        if base.ndim == 0:
            raise ValueError("Cannot index scalar data expression")
        index = _evaluate_data_index_expr(node.index, data)
        _validate_index_value(index, base.shape[0])
        return base[index]
    raise TypeError(f"Cannot evaluate index expression node: {type(node).__name__}")


def _validate_index_value(index: jax.Array, base_size: int) -> None:
    """Validate an integer first-axis index array against one base size."""
    if not jnp.issubdtype(index.dtype, jnp.integer):
        raise TypeError("Index data must be integer")
    if bool(jnp.any((index < 0) | (index >= base_size))):
        raise ValueError(f"Index data out of bounds for axis 0 with size {base_size}")


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
    if isinstance(value, DeferredUnaryOp):
        return UnaryOp(
            value.function,
            _resolve_declaration_expr(value.operand, symbols),
        )
    if isinstance(value, DeferredIndexOp):
        return IndexOp(
            _resolve_declaration_expr(value.base, symbols),
            _resolve_declaration_expr(value.index, symbols),
        )
    raise TypeError(f"Cannot resolve {type(value).__name__} as a declaration expression")


def _is_declaration_expr(value: object) -> bool:
    """Return whether ``value`` can resolve to final expression IR."""
    return isinstance(
        value, Param | Data | DeferredBinOp | DeferredIndexOp | DeferredUnaryOp | int | float
    )


def _is_final_expr_node(value: object) -> bool:
    """Return whether ``value`` is already resolved final expression IR."""
    return isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp)


def _resolve_symbol(symbol: DeclarationSymbol, symbols: SymbolTable) -> str:
    name = symbols.get(symbol)
    if name is None:
        raise ValueError(f"Unknown declaration symbol: {symbol}")
    return name
