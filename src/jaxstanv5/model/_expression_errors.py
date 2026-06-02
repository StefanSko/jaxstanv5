"""Shared diagnostics for model declaration expression values."""

from __future__ import annotations

from collections.abc import Sequence

_ARRAY_LIKE_CONSTANT_MESSAGE = (
    "Array-like constants are not supported in model declaration expressions. "
    "Use Python scalar literals for scalar constants; declare fixed non-scalar inputs "
    "with Data.vector(...), Data.matrix(...), or Data.array(...) and pass them to bind(...)."
)

_NON_SCALAR_DISTRIBUTION_PARAMETER_MESSAGE = (
    "Non-scalar distribution parameters in model declarations must be declared as Data. "
    "Use Data.vector(...), Data.matrix(...), or Data.array(...) and pass values to "
    "bind(...) or simulate_prior_predictive(..., data=...)."
)


def is_array_like_constant(value: object) -> bool:
    """Return whether ``value`` looks like a fixed array-like expression constant."""
    if isinstance(value, str | bytes):
        return False
    if isinstance(value, list):
        return True
    return hasattr(value, "shape") and hasattr(value, "dtype")


def is_non_scalar_array_like_constant(value: object) -> bool:
    """Return whether ``value`` is a non-scalar fixed array-like value."""
    if isinstance(value, str | bytes):
        return False
    if isinstance(value, Sequence):
        return True
    shape = getattr(value, "shape", None)
    if shape is None or not hasattr(value, "dtype"):
        return False
    return tuple(shape) != ()


def array_like_constant_error() -> TypeError:
    """Return the public declaration-language error for array-like constants."""
    return TypeError(_ARRAY_LIKE_CONSTANT_MESSAGE)


def non_scalar_distribution_parameter_error() -> TypeError:
    """Return the public error for hidden non-scalar distribution parameters."""
    return TypeError(_NON_SCALAR_DISTRIBUTION_PARAMETER_MESSAGE)
