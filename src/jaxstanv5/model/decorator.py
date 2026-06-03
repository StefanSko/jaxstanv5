"""Model declaration resolution and binding."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions.core import DiscreteDistribution, Distribution
from jaxstanv5.model._data_schema import (
    DataDimRef,
    DataDimSymbol,
    DataRankSchema,
    DataShapeSchema,
    ResolvedDataRankSchema,
    ResolvedDataSchema,
    ResolvedDataShapeDim,
    ResolvedDataShapeSchema,
)
from jaxstanv5.model._deferred import (
    DeclarationSymbol,
    DeferredBinOp,
    DeferredIndexOp,
    DeferredUnaryOp,
    is_deferred_expr,
)
from jaxstanv5.model._expression_errors import (
    array_like_constant_error,
    is_array_like_constant,
    is_non_scalar_array_like_constant,
    non_scalar_distribution_parameter_error,
)
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    ExprNode,
    IndexOp,
    ParamRef,
    UnaryOp,
    VectorScatterOp,
)

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
    "neg": jnp.negative,
    "sigmoid": jax.nn.sigmoid,
}


@dataclass(frozen=True)
class ResolvedData:
    """Data metadata after declaration symbols are resolved to names."""

    schema: ResolvedDataSchema


@dataclass(frozen=True)
class ResolvedFreeValue:
    """Free NUTS coordinate metadata after declaration symbols are resolved."""

    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedParam:
    """Parameter declaration metadata after declaration symbols are resolved."""

    distribution: Distribution
    constraint: Constraint | None
    size: DataRef | int | None


@dataclass(frozen=True)
class ResolvedStochasticSite:
    """One log-density factor evaluated at a resolved model value expression."""

    name: str
    distribution: Distribution
    value: ExprNode


@dataclass(frozen=True)
class ResolvedObserved:
    """Observed likelihood metadata after declaration symbols are resolved to names."""

    name: str
    distribution: Distribution


@dataclass(frozen=True)
class ModelMeta:
    """Final model metadata attached by ``@model``."""

    params: dict[str, ResolvedParam]
    data: dict[str, ResolvedData]
    observed_nodes: tuple[ResolvedObserved, ...]
    expressions: dict[str, ExprNode]
    free_values: dict[str, ResolvedFreeValue] = field(default_factory=dict)
    stochastic_sites: tuple[ResolvedStochasticSite, ...] = ()


@dataclass(frozen=True)
class _ResolvedDeclarations:
    """Resolved top-level declarations from a declaration class."""

    params: dict[str, ResolvedParam]
    data: dict[str, ResolvedData]
    observed_nodes: tuple[ResolvedObserved, ...]
    free_values: dict[str, ResolvedFreeValue]
    stochastic_sites: tuple[ResolvedStochasticSite, ...]


def _resolve_model_declaration(cls: ModelClass) -> ModelMeta:
    """Resolve a declaration class into final model metadata."""
    symbols = _collect_declaration_symbols(cls)
    declarations = _resolve_declarations(cls, symbols)
    expressions = _resolve_expressions(cls, symbols)

    return ModelMeta(
        params=declarations.params,
        data=declarations.data,
        observed_nodes=declarations.observed_nodes,
        expressions=expressions,
        free_values=declarations.free_values,
        stochastic_sites=declarations.stochastic_sites,
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
    data: dict[str, ResolvedData] = {}
    observed_nodes: list[ResolvedObserved] = []
    free_values: dict[str, ResolvedFreeValue] = {}
    stochastic_sites: list[ResolvedStochasticSite] = []

    for name, value in cls.__dict__.items():
        if isinstance(value, Param):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            if isinstance(distribution, DiscreteDistribution):
                raise TypeError(
                    "Discrete distributions cannot be used as Param priors; "
                    "use them for Observed likelihoods or marginalize discrete latents"
                )
            size = _resolve_declaration_size(value.size, symbols)
            params[name] = ResolvedParam(
                distribution=distribution,
                constraint=value.constraint,
                size=size,
            )
            free_values[name] = ResolvedFreeValue(
                constraint=value.constraint,
                size=size,
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=ParamRef(name),
                )
            )
        elif isinstance(value, Data):
            data[name] = ResolvedData(_resolve_data_schema(value.schema, symbols))
        elif isinstance(value, Observed):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            observed_nodes.append(
                ResolvedObserved(
                    name=name,
                    distribution=distribution,
                )
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=DataRef(name),
                )
            )

    if not params and not observed_nodes:
        raise ValueError("Model declarations must contain at least one stochastic declaration")

    return _ResolvedDeclarations(
        params=params,
        data=data,
        observed_nodes=tuple(observed_nodes),
        free_values=free_values,
        stochastic_sites=tuple(stochastic_sites),
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

        expected = set(meta.data)
        expected.update(observed.name for observed in meta.observed_nodes)
        actual = set(values)
        missing = expected - actual
        extra = actual - expected
        if missing:
            raise ValueError(f"Missing model data: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected model data: {sorted(extra)}")

        data = {name: jnp.asarray(value) for name, value in values.items()}
        _validate_declared_data_values(meta, {name: data[name] for name in meta.data})
        param_shapes = {
            name: _resolve_param_shape(value.size, data)
            for name, value in _resolved_free_values(meta).items()
        }
        n_params = sum(_param_count(shape) for shape in param_shapes.values())
        _validate_bound_index_expressions(meta, data, param_shapes)
        return BoundModel(meta=meta, data=data, param_shapes=param_shapes, n_params=n_params)

    return bind


def _resolved_free_values(meta: ModelMeta) -> dict[str, ResolvedFreeValue]:
    """Return free NUTS values, deriving legacy metadata when absent."""
    if meta.free_values:
        return meta.free_values
    return {
        name: ResolvedFreeValue(constraint=param.constraint, size=param.size)
        for name, param in meta.params.items()
    }


def _resolved_stochastic_sites(meta: ModelMeta) -> tuple[ResolvedStochasticSite, ...]:
    """Return log-density sites, deriving legacy metadata when absent."""
    if meta.stochastic_sites:
        return meta.stochastic_sites
    param_sites = tuple(
        ResolvedStochasticSite(
            name=name,
            distribution=param.distribution,
            value=ParamRef(name),
        )
        for name, param in meta.params.items()
    )
    observed_sites = tuple(
        ResolvedStochasticSite(
            name=observed.name,
            distribution=observed.distribution,
            value=DataRef(observed.name),
        )
        for observed in meta.observed_nodes
    )
    return param_sites + observed_sites


def _normalize_declared_data_values(
    meta: ModelMeta,
    values: Mapping[str, object] | None,
) -> dict[str, jax.Array]:
    """Normalize and validate values for declared ``Data`` nodes only."""
    raw_values: Mapping[str, object] = {} if values is None else values
    expected = set(meta.data)
    actual = set(raw_values)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise ValueError(f"Missing model data: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected model data: {sorted(extra)}")

    data = {name: jnp.asarray(value) for name, value in raw_values.items()}
    _validate_declared_data_values(meta, data)
    return data


def _validate_declared_data_values(meta: ModelMeta, data: dict[str, jax.Array]) -> None:
    """Validate bound data arrays against resolved declaration schemas."""
    for name, resolved in meta.data.items():
        value = data[name]
        schema = resolved.schema
        if isinstance(schema, ResolvedDataRankSchema):
            if value.ndim != schema.rank:
                raise ValueError(
                    f"Data {name!r} has wrong rank: expected {schema.rank}, got {value.ndim}"
                )
        elif isinstance(schema, ResolvedDataShapeSchema):
            expected_shape = tuple(_resolve_data_shape_dim(dim, data) for dim in schema.dims)
            if value.shape != expected_shape:
                raise ValueError(
                    f"Data {name!r} has wrong shape: expected {expected_shape}, got {value.shape}"
                )
        else:
            raise TypeError(f"Unknown data schema for {name!r}: {type(schema).__name__}")


def _resolve_data_shape_dim(dim: ResolvedDataShapeDim, data: dict[str, jax.Array]) -> int:
    """Resolve one data shape dimension against concrete data values."""
    if isinstance(dim, int):
        return _validate_parameter_size(dim, "Data shape dimension")

    dim_value = data[dim.name]
    if dim_value.ndim != 0:
        raise ValueError(f"Data shape dimension {dim.name!r} must be scalar")
    if not jnp.issubdtype(dim_value.dtype, jnp.integer):
        raise TypeError(f"Data shape dimension {dim.name!r} must be integer")
    return _validate_parameter_size(int(dim_value), f"Data shape dimension {dim.name!r}")


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
    for site in _resolved_stochastic_sites(meta):
        _validate_distribution_index_expressions(site.distribution, data, param_shapes)
        _validate_index_expr(site.value, data, param_shapes)
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

    for distribution_field in fields(distribution):
        value = getattr(distribution, distribution_field.name)
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
    elif isinstance(node, VectorScatterOp):
        _validate_index_expr(node.length, data, param_shapes)
        _validate_index_expr(node.observed_idx, data, param_shapes)
        _validate_index_expr(node.observed_values, data, param_shapes)
        _validate_index_expr(node.missing_idx, data, param_shapes)
        _validate_index_expr(node.missing_values, data, param_shapes)
        _validate_vector_scatter_op(node, data, param_shapes)


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
    if isinstance(node, VectorScatterOp):
        length = _validate_vector_scatter_op(node, data, param_shapes)
        return (length,)
    raise TypeError(f"Cannot infer shape for expression node: {type(node).__name__}")


def _validate_vector_scatter_op(
    node: VectorScatterOp,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> int:
    """Validate one partial-observed vector assembly expression."""
    length = _resolve_vector_scatter_length(node.length, data)
    observed_idx = _evaluate_data_index_expr(node.observed_idx, data)
    missing_idx = _evaluate_data_index_expr(node.missing_idx, data)

    if observed_idx.ndim != 1 or missing_idx.ndim != 1:
        raise ValueError("Partial-observed indexes must be rank-1 vectors")
    _validate_index_value(observed_idx, length)
    _validate_index_value(missing_idx, length)

    observed_shape = _infer_expr_shape(node.observed_values, data, param_shapes)
    missing_shape = _infer_expr_shape(node.missing_values, data, param_shapes)
    if observed_shape != (observed_idx.shape[0],):
        raise ValueError("Partial-observed values must match observed_idx length")
    if missing_shape != (missing_idx.shape[0],):
        raise ValueError("Partial-observed free values must match missing_idx length")

    observed_positions = tuple(int(value) for value in observed_idx.tolist())
    missing_positions = tuple(int(value) for value in missing_idx.tolist())
    observed_set = set(observed_positions)
    missing_set = set(missing_positions)
    if len(observed_set) != len(observed_positions) or len(missing_set) != len(missing_positions):
        raise ValueError("Partial-observed indexes must not contain duplicates")
    if observed_set & missing_set:
        raise ValueError("Partial-observed indexes must be disjoint")
    if observed_set | missing_set != set(range(length)):
        raise ValueError("Partial-observed indexes must cover every vector position exactly once")
    return length


def _resolve_vector_scatter_length(node: ExprNode, data: dict[str, jax.Array]) -> int:
    length_value = _evaluate_data_index_expr(node, data)
    if length_value.ndim != 0:
        raise ValueError("Partial-observed vector length must be scalar")
    if not jnp.issubdtype(length_value.dtype, jnp.integer):
        raise TypeError("Partial-observed vector length must be integer")
    return _validate_parameter_size(int(length_value), "Partial-observed vector length")


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
    if isinstance(node, VectorScatterOp):
        raise TypeError("Partial-observed vector assembly cannot be used as an index expression")
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


def _resolve_data_schema(
    schema: DataRankSchema | DataShapeSchema, symbols: SymbolTable
) -> ResolvedDataSchema:
    """Resolve a data declaration schema into named metadata."""
    if isinstance(schema, DataRankSchema):
        return ResolvedDataRankSchema(schema.rank)
    return ResolvedDataShapeSchema(
        tuple(_resolve_data_shape_schema_dim(dim, symbols) for dim in schema.dims)
    )


def _resolve_data_shape_schema_dim(
    dim: int | DataDimSymbol,
    symbols: SymbolTable,
) -> ResolvedDataShapeDim:
    if isinstance(dim, int):
        return dim
    return DataDimRef(_resolve_symbol(dim.symbol, symbols))


def _resolve_declaration_size(size: object, symbols: SymbolTable) -> DataRef | int | None:
    """Resolve a declaration-size value into final size metadata."""
    if size is None:
        return None
    if isinstance(size, int):
        return _validate_parameter_size(size, "Parameter size")
    if isinstance(size, Data):
        if isinstance(size.schema, DataShapeSchema) and size.schema.dims == ():
            return DataRef(_resolve_symbol(size.symbol, symbols))
        if isinstance(size.schema, DataRankSchema) and size.schema.rank == 0:
            return DataRef(_resolve_symbol(size.symbol, symbols))
        raise TypeError("Data-dependent parameter sizes must use scalar data declarations")
    raise TypeError(f"Cannot resolve {type(size).__name__} as a declaration size")


def _resolve_declaration_distribution(
    distribution: Distribution,
    symbols: SymbolTable,
) -> Distribution:
    """Resolve symbolic distribution fields into final expression nodes."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution
    resolved = {
        distribution_field.name: _resolve_declaration_distribution_field(
            getattr(distribution, distribution_field.name),
            symbols,
        )
        for distribution_field in fields(distribution)
    }
    return type(distribution)(**resolved)


def _resolve_declaration_distribution_field(value: object, symbols: SymbolTable) -> object:
    if _is_declaration_expr(value):
        return _resolve_declaration_expr(value, symbols)
    if _is_final_expr_node(value):
        raise TypeError("Final expression nodes are not valid in model declarations")
    if is_dataclass(value) and not isinstance(value, type):
        return _resolve_declaration_distribution(cast(Distribution, value), symbols)
    if is_non_scalar_array_like_constant(value):
        raise non_scalar_distribution_parameter_error()
    return value


def _resolve_declaration_expr(value: object, symbols: SymbolTable) -> ExprNode:
    """Resolve class-body declaration syntax into final expression IR."""
    if isinstance(value, Param):
        return ParamRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, Data):
        return DataRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, int | float):
        return ConstNode(value)
    if is_array_like_constant(value):
        raise array_like_constant_error()
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
    return isinstance(
        value, ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp | VectorScatterOp
    )


def _resolve_symbol(symbol: DeclarationSymbol, symbols: SymbolTable) -> str:
    name = symbols.get(symbol)
    if name is None:
        raise ValueError(f"Unknown declaration symbol: {symbol}")
    return name
