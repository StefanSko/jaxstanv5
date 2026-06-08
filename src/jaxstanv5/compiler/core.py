"""Core compiler — evaluate symbolic model IR to JAX-callable functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import cast

import jax
import jax.numpy as jnp

from jaxstanv5.distributions.core import Distribution
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, _resolved_free_values, _resolved_stochastic_sites
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

_BINOPS: dict[str, Callable[[jax.Array, jax.Array], jax.Array]] = {
    "+": jnp.add,
    "-": jnp.subtract,
    "*": jnp.multiply,
    "/": jnp.divide,
}

_UNARY_FUNCTIONS: dict[str, Callable[[jax.Array], jax.Array]] = {
    "exp": jnp.exp,
    "neg": jnp.negative,
    "sigmoid": jax.nn.sigmoid,
}

type EvaluatedIndexAtom = jax.Array | slice
type EvaluatedIndex = EvaluatedIndexAtom | tuple[EvaluatedIndexAtom, ...]


def _evaluate_index_spec(spec: IndexSpec, values: dict[str, jax.Array]) -> EvaluatedIndex:
    """Evaluate explicit index IR to a JAX-compatible index object."""
    if isinstance(spec, ScalarIndex):
        return _evaluate_expr(spec.expr, values)
    if isinstance(spec, FullSlice):
        return slice(None)
    if isinstance(spec, IndexTuple):
        items: list[EvaluatedIndexAtom] = []
        for item in spec.items:
            evaluated = _evaluate_index_spec(item, values)
            if isinstance(evaluated, tuple):
                raise TypeError("Nested index tuples are not supported")
            items.append(evaluated)
        return tuple(items)
    raise TypeError(f"Cannot evaluate index spec: {type(spec).__name__}")


def _evaluate_expr(node: ExprNode, values: dict[str, jax.Array]) -> jax.Array:
    """Evaluate a symbolic expression tree to a concrete JAX array."""
    if isinstance(node, ParamRef):
        return values[node.name]
    if isinstance(node, DataRef):
        return values[node.name]
    if isinstance(node, ConstNode):
        return jnp.asarray(node.value)
    if isinstance(node, BinOp):
        left = _evaluate_expr(node.left, values)
        right = _evaluate_expr(node.right, values)
        op_fn = _BINOPS.get(node.op)
        if op_fn is None:
            raise ValueError(f"Unknown binary operator: {node.op!r}")
        return op_fn(left, right)
    if isinstance(node, UnaryOp):
        operand = _evaluate_expr(node.operand, values)
        function = _UNARY_FUNCTIONS.get(node.function)
        if function is None:
            raise ValueError(f"Unknown unary function: {node.function!r}")
        return function(operand)
    if isinstance(node, IndexOp):
        base = _evaluate_expr(node.base, values)
        index = _evaluate_index_spec(node.index, values)
        return base[index]
    if isinstance(node, VectorScatterOp):
        observed_idx = _evaluate_expr(node.observed_idx, values)
        observed_values = _evaluate_expr(node.observed_values, values)
        missing_idx = _evaluate_expr(node.missing_idx, values)
        missing_values = _evaluate_expr(node.missing_values, values)
        length = observed_idx.shape[0] + missing_idx.shape[0]
        dtype = jnp.result_type(observed_values, missing_values)
        result = jnp.zeros((length,), dtype=dtype)
        return result.at[observed_idx].set(observed_values).at[missing_idx].set(missing_values)
    raise TypeError(f"Cannot evaluate expression node: {type(node).__name__}")


def _is_expr_node(value: object) -> bool:
    """Return whether ``value`` is a final expression IR node."""
    return isinstance(
        value, ParamRef | DataRef | ConstNode | BinOp | IndexOp | UnaryOp | VectorScatterOp
    )


def _evaluate_distribution[DistributionT: Distribution](
    distribution: DistributionT,
    values: dict[str, jax.Array],
) -> DistributionT:
    """Evaluate expression fields in a distribution to concrete JAX arrays."""
    if not is_dataclass(distribution) or isinstance(distribution, type):
        return distribution

    resolved: dict[str, object] = {}
    for f in fields(distribution):
        val = getattr(distribution, f.name)
        if _is_expr_node(val):
            resolved[f.name] = _evaluate_expr(val, values)
        elif is_dataclass(val) and not isinstance(val, type):
            resolved[f.name] = _evaluate_distribution(val, values)
        else:
            resolved[f.name] = val

    return cast(DistributionT, type(distribution)(**resolved))


def _param_vector_size(shapes: dict[str, tuple[int, ...]]) -> int:
    """Return the flat unconstrained size implied by parameter shapes."""
    total = 0
    for shape in shapes.values():
        size = 1
        for dim in shape:
            size *= dim
        total += size
    return total


def _validate_flat_param_vector(flat: jax.Array, shapes: dict[str, tuple[int, ...]]) -> None:
    """Validate the flat unconstrained parameter vector shape."""
    if flat.ndim != 1:
        raise ValueError("Unconstrained parameter vector q must be one-dimensional")

    expected = _param_vector_size(shapes)
    actual = flat.shape[0]
    if actual != expected:
        raise ValueError(
            f"Unconstrained parameter vector q has wrong length: expected {expected}, got {actual}"
        )


def _split_params(
    flat: jax.Array,
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, jax.Array]:
    """Split a flat unconstrained parameter vector into named segments."""
    _validate_flat_param_vector(flat, shapes)

    result: dict[str, jax.Array] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 1
        for d in shape:
            size *= d
        if shape == ():
            result[name] = flat[offset]
        else:
            result[name] = flat[offset : offset + size].reshape(shape)
        offset += size
    return result


def _constrain_params(
    params: dict[str, jax.Array],
    meta: ModelMeta,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Apply constraint transforms and return constrained params + log-Jacobian sum."""
    constrained: dict[str, jax.Array] = {}
    log_jac: jax.Array = jnp.array(0.0)
    for name, value in _resolved_free_values(meta).items():
        val = params[name]
        if value.constraint is not None:
            constrained[name] = cast(jax.Array, value.constraint.inverse_transform(val))
            log_jac = log_jac + jnp.sum(value.constraint.log_abs_det_jacobian(val))
        else:
            constrained[name] = val
    return constrained, log_jac


def _build_log_density(bound: BoundModel) -> Callable[[jax.Array], jax.Array]:
    """Build a JAX-traceable log-density closure from static bound-model metadata."""
    meta = bound.meta
    shapes = bound.param_shapes

    def log_prob(q: jax.Array) -> jax.Array:
        params = _split_params(q, shapes)
        constrained, log_jac = _constrain_params(params, meta)
        values = {**constrained, **bound.data}

        lp: jax.Array = log_jac

        for site in _resolved_stochastic_sites(meta):
            dist = _evaluate_distribution(site.distribution, values)
            value = _evaluate_expr(site.value, values)
            lp = lp + jnp.sum(dist.log_prob(value))

        return lp

    return log_prob


def compile_log_density(bound: BoundModel) -> Callable[[jax.Array], jax.Array]:
    """Compile a bound model into a lazily JIT-compiled log-density function.

    Returns a function ``f(q) -> scalar`` where ``q`` is a flat unconstrained
    parameter vector.  The first same-shape call traces and compiles the numeric
    log-density; subsequent same-shape calls reuse the compiled executable.
    """
    return jax.jit(_build_log_density(bound))
