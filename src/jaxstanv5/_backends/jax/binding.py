"""JAX data binding and bind-time validation for resolved model metadata."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp

from jaxstanv5._backends.jax.distributions import (
    batch_shape as distribution_batch_shape,
)
from jaxstanv5._backends.jax.distributions import (
    event_shape as distribution_event_shape,
)
from jaxstanv5._backends.jax.distributions import (
    is_sampleable,
    validate_scale_tril,
)
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions.core import DiscreteDistribution, Distribution
from jaxstanv5.distributions.counts import (
    Bernoulli,
    BetaBinomial,
    Binomial,
    NegativeBinomial,
    Poisson,
)
from jaxstanv5.distributions.multivariate import MultivariateNormal
from jaxstanv5.distributions.ordinal import OrderedLogistic
from jaxstanv5.model._data_schema import (
    ResolvedDataRankSchema,
    ResolvedDataShapeDim,
    ResolvedDataShapeSchema,
)
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import (
    ModelMeta,
    ResolvedFreeValue,
    _is_final_expr_node,
    _resolved_free_values,
    _resolved_stochastic_sites,
    _validate_parameter_size,
)
from jaxstanv5.model.dimensions import ResolvedModelDimensions
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

type EvaluatedIndexAtom = jax.Array | slice
type EvaluatedIndex = EvaluatedIndexAtom | tuple[EvaluatedIndexAtom, ...]

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


def bind_model_meta(
    meta: ModelMeta,
    values: Mapping[str, object],
    *,
    dimensions: ResolvedModelDimensions | None = None,
) -> BoundModel:
    """Bind concrete data values to resolved model metadata with the JAX backend."""
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
    _validate_finite_bound_values(data)
    _validate_declared_data_values(meta, {name: data[name] for name in meta.data})
    param_shapes = {
        name: _resolve_param_shape(value.size, data)
        for name, value in _resolved_free_values(meta).items()
    }
    n_params = sum(_param_count(shape) for shape in param_shapes.values())
    _validate_bound_index_expressions(meta, data, param_shapes)
    _validate_stochastic_site_shapes(meta, data, param_shapes)
    _validate_observed_discrete_values(meta, data, param_shapes)
    _validate_bound_distribution_parameters(meta, data, param_shapes)
    if dimensions is not None:
        _validate_bound_dimension_metadata(dimensions, data, param_shapes)
    return BoundModel(
        meta=meta,
        data=data,
        param_shapes=param_shapes,
        n_params=n_params,
        dimensions=dimensions,
    )


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


def _validate_bound_dimension_metadata(
    dimensions: ResolvedModelDimensions,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate bound variable ranks and coordinate lengths against dimension metadata."""
    for variable_name, variable_dims in dimensions.variables.items():
        shape = _bound_variable_shape(variable_name, data, param_shapes)
        if len(variable_dims.names) != len(shape):
            raise ValueError(
                f"Dimension metadata for variable {variable_name!r} has dimension rank "
                f"{len(variable_dims.names)}, but bound value has rank {len(shape)}"
            )
        for dim_name, axis_size in zip(variable_dims.names, shape, strict=True):
            coords = dimensions.coords.get(dim_name)
            if coords is None:
                continue
            if len(coords) != axis_size:
                raise ValueError(
                    f"Dimension {dim_name!r} coordinate length {len(coords)} does not match "
                    f"bound axis size {axis_size} for variable {variable_name!r}"
                )


def _bound_variable_shape(
    variable_name: str,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    param_shape = param_shapes.get(variable_name)
    if param_shape is not None:
        return param_shape
    value = data.get(variable_name)
    if value is not None:
        return value.shape
    raise ValueError(f"Dimension metadata refers to unknown variable {variable_name!r}")


def _validate_finite_bound_values(data: dict[str, jax.Array]) -> None:
    """Reject NaN or infinite bound data before compilation/sampling."""
    for name, value in data.items():
        if bool(jnp.any(jnp.isnan(value))):
            raise ValueError(
                f"Bound value {name!r} contains NaN; use PartialVector.from_nan(...) "
                "and PartiallyObserved for missing values"
            )
        if bool(jnp.any(~jnp.isfinite(value))):
            raise ValueError(f"Bound value {name!r} contains non-finite values")


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


def _validate_stochastic_site_shapes(
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Reject stochastic-site broadcasting that would expand the site value."""
    for site in _resolved_stochastic_sites(meta):
        value_shape = _infer_expr_shape(site.value, data, param_shapes)
        distribution_shape = _distribution_value_shape(site.distribution, data, param_shapes)
        if distribution_shape is None:
            continue
        try:
            broadcast_shape = jnp.broadcast_shapes(value_shape, distribution_shape)
        except ValueError as exc:
            raise ValueError(
                f"Stochastic site {site.name!r} value shape {value_shape} is incompatible "
                f"with distribution shape {distribution_shape}"
            ) from exc
        if broadcast_shape != value_shape:
            raise ValueError(
                f"Stochastic site {site.name!r} value shape {value_shape} would broadcast to "
                f"{broadcast_shape} against distribution shape {distribution_shape}; declare an "
                "explicit matching size or data shape"
            )


def _distribution_value_shape(
    distribution: Distribution,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...] | None:
    """Return batch+event shape for sampleable distributions, or None if unknown."""
    shaped_distribution = _shape_stub_distribution(distribution, data, param_shapes)
    if not is_sampleable(shaped_distribution):
        return None
    return distribution_batch_shape(shaped_distribution) + distribution_event_shape(
        shaped_distribution
    )


def _shape_stub_distribution[DistributionT: Distribution](
    distribution: DistributionT,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> DistributionT:
    """Replace expression fields by zero arrays with inferred bind-time shapes."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution
    resolved: dict[str, object] = {}
    for distribution_field in fields(distribution):
        value = getattr(distribution, distribution_field.name)
        if _is_final_expr_node(value):
            shape = _infer_expr_shape(cast(ExprNode, value), data, param_shapes)
            resolved[distribution_field.name] = jnp.zeros(shape)
        elif is_dataclass(value) and not isinstance(value, type):
            resolved[distribution_field.name] = _shape_stub_distribution(
                cast(Distribution, value),
                data,
                param_shapes,
            )
        else:
            resolved[distribution_field.name] = value
    return cast(DistributionT, type(distribution)(**resolved))


def _validate_observed_discrete_values(
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate concrete observed values for discrete likelihoods."""
    for site in _resolved_stochastic_sites(meta):
        if not isinstance(site.distribution, DiscreteDistribution):
            continue
        value = _evaluate_data_index_expr(site.value, data)
        _validate_integer_observed_value(site.name, value)
        _validate_discrete_observed_support(site.name, site.distribution, value, data, param_shapes)


def _validate_integer_observed_value(name: str, value: jax.Array) -> None:
    if bool(jnp.any(value != jnp.floor(value))):
        raise ValueError(f"Observed site {name!r} contains non-integer values")


def _validate_discrete_observed_support(
    name: str,
    distribution: Distribution,
    value: jax.Array,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    if isinstance(distribution, Bernoulli):
        _validate_value_range(name, value, low=0.0, high=1.0)
    elif isinstance(distribution, Poisson | NegativeBinomial):
        _validate_value_range(name, value, low=0.0, high=None)
    elif isinstance(distribution, Binomial | BetaBinomial):
        _validate_value_range(name, value, low=0.0, high=None)
        total_count = _evaluate_optional_data_expr(distribution.total_count, data)
        if total_count is not None:
            _validate_value_array_upper_bound(name, value, total_count)
    elif isinstance(distribution, OrderedLogistic):
        _validate_value_range(name, value, low=0.0, high=None)
        cutpoint_shape = _distribution_field_shape(distribution.cutpoints, data, param_shapes)
        if cutpoint_shape:
            _validate_value_range(name, value, low=0.0, high=float(cutpoint_shape[-1]))


def _distribution_field_shape(
    value: object,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> tuple[int, ...]:
    if _is_final_expr_node(value):
        return _infer_expr_shape(cast(ExprNode, value), data, param_shapes)
    return jnp.asarray(value).shape


def _validate_value_range(
    name: str,
    value: jax.Array,
    *,
    low: float,
    high: float | None,
) -> None:
    below = value < low
    above = jnp.zeros(value.shape, dtype=bool) if high is None else value > high
    if bool(jnp.any(below | above)):
        if high is None:
            raise ValueError(f"Observed site {name!r} values must be >= {low}")
        raise ValueError(f"Observed site {name!r} values must be between {low} and {high}")


def _validate_value_array_upper_bound(name: str, value: jax.Array, upper: jax.Array) -> None:
    if bool(jnp.any(value > upper)):
        raise ValueError(f"Observed site {name!r} values must be <= total_count")


def _validate_bound_distribution_parameters(
    meta: ModelMeta,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate concrete and safely symbolic distribution parameters at bind time."""
    free_values = _resolved_free_values(meta)
    for site in _resolved_stochastic_sites(meta):
        _validate_bound_distribution_parameter(
            site.name,
            site.distribution,
            data,
            param_shapes,
            free_values,
        )


def _validate_bound_distribution_parameter(
    site_name: str,
    distribution: Distribution,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
    free_values: dict[str, ResolvedFreeValue],
) -> None:
    """Validate one distribution's bind-time parameters."""
    if isinstance(distribution, MultivariateNormal) and not _is_valid_mvn_scale_tril_expr(
        distribution.scale_tril,
        data,
        param_shapes,
        free_values,
        name=f"{site_name!r} scale_tril",
        top_level=True,
    ):
        raise TypeError(
            f"MultivariateNormal scale_tril for site {site_name!r} must be a validated "
            "Cholesky factor, optionally multiplied or divided by a positive scalar"
        )
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return
    for distribution_field in fields(distribution):
        value = getattr(distribution, distribution_field.name)
        if is_dataclass(value) and not isinstance(value, type):
            _validate_bound_distribution_parameter(
                site_name,
                cast(Distribution, value),
                data,
                param_shapes,
                free_values,
            )


def _is_valid_mvn_scale_tril_expr(
    value: object,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
    free_values: dict[str, ResolvedFreeValue],
    *,
    name: str,
    top_level: bool,
) -> bool:
    """Return whether an MVN scale expression is valid by construction."""
    concrete = _evaluate_optional_data_expr(value, data)
    if concrete is not None:
        if not top_level and concrete.ndim < 2:
            return False
        validate_scale_tril(concrete, name=name)
        return True

    if not isinstance(value, BinOp):
        return False

    if value.op == "*":
        return (
            _is_valid_mvn_scale_tril_expr(
                value.left,
                data,
                param_shapes,
                free_values,
                name=name,
                top_level=False,
            )
            and _is_positive_scalar_expr(value.right, data, param_shapes, free_values)
        ) or (
            _is_valid_mvn_scale_tril_expr(
                value.right,
                data,
                param_shapes,
                free_values,
                name=name,
                top_level=False,
            )
            and _is_positive_scalar_expr(value.left, data, param_shapes, free_values)
        )
    if value.op == "/":
        return _is_valid_mvn_scale_tril_expr(
            value.left,
            data,
            param_shapes,
            free_values,
            name=name,
            top_level=False,
        ) and _is_positive_scalar_expr(value.right, data, param_shapes, free_values)
    return False


def _is_positive_scalar_expr(
    value: object,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
    free_values: dict[str, ResolvedFreeValue],
) -> bool:
    """Return whether an expression is provably scalar and strictly positive."""
    if _is_final_expr_node(value):
        shape = _infer_expr_shape(cast(ExprNode, value), data, param_shapes)
        if shape != ():
            return False
    elif jnp.asarray(value).shape != ():
        return False

    concrete = _evaluate_optional_data_expr(value, data)
    if concrete is not None:
        return bool(concrete > 0.0)

    if isinstance(value, ParamRef):
        free_value = free_values.get(value.name)
        return free_value is not None and isinstance(free_value.constraint, Positive)
    if isinstance(value, UnaryOp):
        return value.function in {"exp", "sigmoid"}
    if isinstance(value, BinOp):
        left_positive = _is_positive_scalar_expr(value.left, data, param_shapes, free_values)
        right_positive = _is_positive_scalar_expr(value.right, data, param_shapes, free_values)
        if value.op in {"*", "/"}:
            return left_positive and right_positive
    return False


def _evaluate_optional_data_expr(value: object, data: dict[str, jax.Array]) -> jax.Array | None:
    """Evaluate a data/constant expression, or return None for parameter-dependent values."""
    if _is_final_expr_node(value):
        try:
            return _evaluate_data_index_expr(cast(ExprNode, value), data)
        except TypeError:
            return None
    return jnp.asarray(value)


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
        _validate_index_spec_exprs(node.index, data, param_shapes)
        _validate_single_index_op(node, data, param_shapes)
    elif isinstance(node, VectorScatterOp):
        _validate_index_expr(node.length, data, param_shapes)
        _validate_index_expr(node.observed_idx, data, param_shapes)
        _validate_index_expr(node.observed_values, data, param_shapes)
        _validate_index_expr(node.missing_idx, data, param_shapes)
        _validate_index_expr(node.missing_values, data, param_shapes)
        _validate_vector_scatter_op(node, data, param_shapes)


def _validate_index_spec_exprs(
    spec: IndexSpec,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate expression subtrees nested inside explicit index IR."""
    if isinstance(spec, ScalarIndex):
        _validate_index_expr(spec.expr, data, param_shapes)
    elif isinstance(spec, FullSlice):
        return
    elif isinstance(spec, IndexTuple):
        for item in spec.items:
            _validate_index_spec_exprs(item, data, param_shapes)
    else:
        raise TypeError(f"Cannot validate index spec: {type(spec).__name__}")


def _infer_indexed_shape(
    base_shape: tuple[int, ...],
    spec: IndexSpec,
    data: dict[str, jax.Array],
) -> tuple[int, ...]:
    """Infer and validate the result shape of supported declaration indexing."""
    if not base_shape:
        raise ValueError("Cannot index scalar expression")

    items = _index_spec_items(spec)
    if not items:
        raise TypeError("Empty index tuples are not supported in model declaration expressions")

    axis = 0
    non_scalar_index_count = 0
    non_scalar_index_position: int | None = None
    non_scalar_output_start: int | None = None
    non_scalar_output_ndim = 0
    scalar_index_positions: list[int] = []
    result_shape: list[int] = []
    for position, item in enumerate(items):
        if axis >= len(base_shape):
            raise ValueError(f"Too many index items for expression with rank {len(base_shape)}")
        if isinstance(item, FullSlice):
            result_shape.append(base_shape[axis])
        elif isinstance(item, ScalarIndex):
            index_value = _evaluate_data_index_expr(item.expr, data)
            _validate_index_value(index_value, base_shape[axis], axis=axis)
            if index_value.ndim == 0:
                scalar_index_positions.append(position)
            else:
                non_scalar_index_count += 1
                if non_scalar_index_count > 1:
                    raise TypeError("Index tuples support at most one non-scalar index expression")
                non_scalar_index_position = position
                non_scalar_output_start = len(result_shape)
                non_scalar_output_ndim = len(index_value.shape)
                result_shape.extend(index_value.shape)
        elif isinstance(item, IndexTuple):
            raise TypeError("Nested index tuples are not supported")
        else:
            raise TypeError(f"Cannot infer shape for index spec: {type(item).__name__}")
        axis += 1

    result_shape.extend(base_shape[axis:])
    if _advanced_index_shape_moves_to_front(
        items,
        scalar_index_positions=tuple(scalar_index_positions),
        non_scalar_index_position=non_scalar_index_position,
    ):
        if non_scalar_output_start is None:
            raise TypeError("Cannot move missing non-scalar index shape")
        non_scalar_output_stop = non_scalar_output_start + non_scalar_output_ndim
        return tuple(
            result_shape[non_scalar_output_start:non_scalar_output_stop]
            + result_shape[:non_scalar_output_start]
            + result_shape[non_scalar_output_stop:]
        )
    return tuple(result_shape)


def _advanced_index_shape_moves_to_front(
    items: tuple[IndexSpec, ...],
    *,
    scalar_index_positions: tuple[int, ...],
    non_scalar_index_position: int | None,
) -> bool:
    """Return whether NumPy/JAX moves the lone array index shape to the front."""
    if non_scalar_index_position is None:
        return False
    for scalar_position in scalar_index_positions:
        start = min(scalar_position, non_scalar_index_position) + 1
        stop = max(scalar_position, non_scalar_index_position)
        if any(isinstance(items[position], FullSlice) for position in range(start, stop)):
            return True
    return False


def _index_spec_items(spec: IndexSpec) -> tuple[IndexSpec, ...]:
    if isinstance(spec, IndexTuple):
        return spec.items
    return (spec,)


def _validate_single_index_op(
    node: IndexOp,
    data: dict[str, jax.Array],
    param_shapes: dict[str, tuple[int, ...]],
) -> None:
    """Validate one concrete index operation against all consumed axes."""
    base_shape = _infer_expr_shape(node.base, data, param_shapes)
    _infer_indexed_shape(base_shape, node.index, data)


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
        return _infer_indexed_shape(base_shape, node.index, data)
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


def _evaluate_data_index_spec(spec: IndexSpec, data: dict[str, jax.Array]) -> EvaluatedIndex:
    """Evaluate explicit index IR that must depend only on data and constants."""
    if isinstance(spec, ScalarIndex):
        return _evaluate_data_index_expr(spec.expr, data)
    if isinstance(spec, FullSlice):
        return slice(None)
    if isinstance(spec, IndexTuple):
        items: list[EvaluatedIndexAtom] = []
        for item in spec.items:
            evaluated = _evaluate_data_index_spec(item, data)
            if isinstance(evaluated, tuple):
                raise TypeError("Nested index tuples are not supported")
            items.append(evaluated)
        return tuple(items)
    raise TypeError(f"Cannot evaluate index spec: {type(spec).__name__}")


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
        _infer_indexed_shape(base.shape, node.index, data)
        index = _evaluate_data_index_spec(node.index, data)
        return base[index]
    if isinstance(node, VectorScatterOp):
        raise TypeError("Partial-observed vector assembly cannot be used as an index expression")
    raise TypeError(f"Cannot evaluate index expression node: {type(node).__name__}")


def _validate_index_value(index: jax.Array, base_size: int, *, axis: int = 0) -> None:
    """Validate an integer index array against one concrete axis size."""
    if not jnp.issubdtype(index.dtype, jnp.integer):
        raise TypeError("Index data must be integer")
    if bool(jnp.any((index < 0) | (index >= base_size))):
        raise ValueError(f"Index data out of bounds for axis {axis} with size {base_size}")


def _param_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dim in shape:
        if dim < 0:
            raise ValueError("Parameter shape dimensions must be non-negative")
        count *= dim
    return count
