"""Model declaration resolution and binding."""

from __future__ import annotations

import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass
from typing import SupportsFloat, cast

from jaxstanv5.constraints import Interval, Positive, UnitInterval
from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions._symbolic_validation import reject_opaque_symbolic_distribution
from jaxstanv5.distributions.continuous import Beta, Exponential, HalfNormal, Uniform
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
from jaxstanv5.model.core import Data, Observed, Param, PartiallyObserved
from jaxstanv5.model.expr import (
    BinOp,
    ConstNode,
    DataRef,
    ExprNode,
    FullSlice,
    IndexOp,
    IndexSpec,
    IndexTuple,
    ParamRef,
    ScalarIndex,
    UnaryOp,
    VectorScatterOp,
)

type ModelClass = type[object]
type SymbolTable = dict[DeclarationSymbol, str]


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
    _reject_declaration_inheritance(cls)
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


def _reject_declaration_inheritance(cls: ModelClass) -> None:
    """Reject base classes so model meaning stays local to one class body."""
    if cls.__bases__ == (object,):
        return
    base_names = ", ".join(repr(base.__name__) for base in cls.__bases__ if base is not object)
    raise TypeError(
        f"Model declaration classes must not use inheritance: {cls.__name__!r} "
        f"inherits from {base_names}. All declarations must live in the decorated "
        "class body; inherited declarations would be silently ignored otherwise"
    )


def _collect_declaration_symbols(cls: ModelClass) -> SymbolTable:
    """Collect declaration symbols and reject declaration aliases."""
    symbols: SymbolTable = {}

    for name, value in cls.__dict__.items():
        if isinstance(value, Param | Data | Observed | PartiallyObserved):
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
            _validate_param_prior_constraint(
                name=name,
                distribution=distribution,
                constraint=value.constraint,
            )
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
        elif isinstance(value, PartiallyObserved):
            distribution = _resolve_declaration_distribution(value.distribution, symbols)
            if isinstance(distribution, DiscreteDistribution):
                raise TypeError(
                    "Discrete distributions cannot be partially observed NUTS values; "
                    "marginalize discrete missing values or impute them posterior-predictively"
                )
            free_values[name] = ResolvedFreeValue(
                constraint=None,
                size=_resolve_partially_observed_missing_size(value, symbols),
            )
            stochastic_sites.append(
                ResolvedStochasticSite(
                    name=name,
                    distribution=distribution,
                    value=_resolve_partially_observed_vector(value, name, symbols),
                )
            )

    if not stochastic_sites:
        raise ValueError("Model declarations must contain at least one stochastic declaration")

    return _ResolvedDeclarations(
        params=params,
        data=data,
        observed_nodes=tuple(observed_nodes),
        free_values=free_values,
        stochastic_sites=tuple(stochastic_sites),
    )


def _validate_param_prior_constraint(
    *,
    name: str,
    distribution: Distribution,
    constraint: Constraint | None,
) -> None:
    """Validate known concrete prior supports against explicit constraints."""
    if isinstance(distribution, Exponential | HalfNormal):
        if not isinstance(constraint, Positive):
            raise TypeError(
                f"Parameter {name!r} prior has support (0, inf); declare "
                f"Param({type(distribution).__name__}(...), constraint=Positive())"
            )
        return

    if isinstance(distribution, Beta):
        if not isinstance(constraint, UnitInterval):
            raise TypeError(
                f"Parameter {name!r} prior has support (0, 1); declare "
                "Param(Beta(...), constraint=UnitInterval())"
            )
        return

    if isinstance(distribution, Uniform):
        bounds = _concrete_uniform_bounds(distribution)
        if bounds is None:
            return
        low, high = bounds
        if _constraint_matches_uniform_support(constraint, low=low, high=high):
            return
        raise TypeError(
            f"Parameter {name!r} Uniform prior has support ({low}, {high}); declare "
            f"Param(Uniform({low}, {high}), constraint=Interval({low}, {high}))"
        )


def _concrete_uniform_bounds(distribution: Uniform) -> tuple[float, float] | None:
    """Return concrete scalar Uniform bounds, or None for symbolic bounds."""
    low = _concrete_scalar_bound(distribution.low)
    high = _concrete_scalar_bound(distribution.high)
    if low is None or high is None:
        return None
    return (low, high)


def _concrete_scalar_bound(value: object) -> float | None:
    """Return a concrete scalar distribution bound after declaration resolution."""
    if isinstance(value, ConstNode):
        return float(value.value)
    if _is_final_expr_node(value):
        return None
    shape = getattr(value, "shape", None)
    if shape is not None:
        if tuple(shape) != ():
            return None
        return float(cast(SupportsFloat, value))
    if isinstance(value, int | float):
        return float(value)
    return None


def _constraint_matches_uniform_support(
    constraint: Constraint | None,
    *,
    low: float,
    high: float,
) -> bool:
    """Return whether a Uniform prior has an explicit compatible constraint."""
    if (
        _same_scalar_bound(low, 0.0)
        and _same_scalar_bound(high, 1.0)
        and isinstance(constraint, UnitInterval)
    ):
        return True
    if isinstance(constraint, Interval):
        return _same_scalar_bound(constraint.lower, low) and _same_scalar_bound(
            constraint.upper, high
        )
    return False


def _same_scalar_bound(left: float, right: float) -> bool:
    """Compare scalar bounds at the active JAX float resolution without importing JAX."""
    if _jax_enable_x64_loaded():
        return float(left) == float(right)
    return _float32(left) == _float32(right)


def _jax_enable_x64_loaded() -> bool:
    """Return the loaded JAX x64 flag without importing JAX on authoring paths."""
    jax_module = sys.modules.get("jax")
    config = getattr(jax_module, "config", None)
    read = getattr(config, "read", None)
    if callable(read):
        return bool(read("jax_enable_x64"))
    return bool(getattr(config, "jax_enable_x64", False))


def _float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", float(value)))[0]


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
        from jaxstanv5._backends.jax.binding import bind_model_meta

        return bind_model_meta(meta, values)

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


def _resolve_partially_observed_missing_size(
    value: PartiallyObserved,
    symbols: SymbolTable,
) -> DataRef | int:
    """Resolve the free-coordinate size from an exact missing-index data schema."""
    schema = value.missing_idx.schema
    if not isinstance(schema, DataShapeSchema) or len(schema.dims) != 1:
        raise TypeError("PartiallyObserved missing_idx must be declared as Data.vector(length)")

    dim = schema.dims[0]
    if isinstance(dim, int):
        return _validate_parameter_size(dim, "PartiallyObserved missing size")
    if isinstance(dim, DataDimSymbol):
        return DataRef(_resolve_symbol(dim.symbol, symbols))
    raise TypeError(f"Unknown partial-observed missing size dimension: {type(dim).__name__}")


def _resolve_partially_observed_vector(
    value: PartiallyObserved,
    name: str,
    symbols: SymbolTable,
) -> VectorScatterOp:
    """Resolve a partially observed declaration into full-vector assembly IR."""
    return VectorScatterOp(
        length=_resolve_declaration_expr(value.length, symbols),
        observed_idx=_resolve_declaration_expr(value.observed_idx, symbols),
        observed_values=_resolve_declaration_expr(value.observed, symbols),
        missing_idx=_resolve_declaration_expr(value.missing_idx, symbols),
        missing_values=ParamRef(name),
    )


def _resolve_declaration_distribution(
    distribution: Distribution,
    symbols: SymbolTable,
) -> Distribution:
    """Resolve symbolic distribution fields into final expression nodes."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        reject_opaque_symbolic_distribution(distribution)
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
    reject_opaque_symbolic_distribution(value)
    return value


def _resolve_index_spec(value: object, symbols: SymbolTable) -> IndexSpec:
    """Resolve raw class-body indexing syntax into explicit final index IR."""
    if isinstance(value, slice):
        return _resolve_slice_index_spec(value)
    if isinstance(value, tuple):
        if not value:
            raise TypeError("Empty index tuples are not supported in model declarations")
        return IndexTuple(tuple(_resolve_index_tuple_item(item, symbols) for item in value))
    if isinstance(value, bool):
        raise TypeError("Index constants must be integers, not bool")
    if isinstance(value, ScalarIndex | FullSlice | IndexTuple):
        raise TypeError("Final index nodes are not valid in model declarations")
    return ScalarIndex(_resolve_declaration_expr(value, symbols))


def _resolve_index_tuple_item(value: object, symbols: SymbolTable) -> IndexSpec:
    if isinstance(value, tuple):
        raise TypeError("Nested index tuples are not supported in model declarations")
    return _resolve_index_spec(value, symbols)


def _resolve_slice_index_spec(value: slice) -> FullSlice:
    if value.start is None and value.stop is None and value.step is None:
        return FullSlice()
    raise TypeError("Only full slices ':' are supported in model declaration indexes")


def _resolve_declaration_expr(value: object, symbols: SymbolTable) -> ExprNode:
    """Resolve class-body declaration syntax into final expression IR."""
    if isinstance(value, Param):
        return ParamRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, Data):
        return DataRef(_resolve_symbol(value.symbol, symbols))
    if isinstance(value, PartiallyObserved):
        return _resolve_partially_observed_vector(
            value,
            _resolve_symbol(value.symbol, symbols),
            symbols,
        )
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
            _resolve_index_spec(value.index, symbols),
        )
    raise TypeError(f"Cannot resolve {type(value).__name__} as a declaration expression")


def _is_declaration_expr(value: object) -> bool:
    """Return whether ``value`` can resolve to final expression IR."""
    return isinstance(
        value,
        Param
        | Data
        | PartiallyObserved
        | DeferredBinOp
        | DeferredIndexOp
        | DeferredUnaryOp
        | int
        | float,
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
