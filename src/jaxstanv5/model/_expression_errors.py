"""Shared diagnostics for model declaration expression values."""

from __future__ import annotations

_ARRAY_LIKE_CONSTANT_MESSAGE = (
    "Array-like constants are not supported in model declaration expressions. "
    "Use Python scalar literals for scalar constants; declare fixed non-scalar inputs "
    "with Data() and pass them to bind(...)."
)


def is_array_like_constant(value: object) -> bool:
    """Return whether ``value`` looks like a fixed array-like expression constant."""
    if isinstance(value, str | bytes):
        return False
    if isinstance(value, list):
        return True
    return hasattr(value, "shape") and hasattr(value, "dtype")


def array_like_constant_error() -> TypeError:
    """Return the public declaration-language error for array-like constants."""
    return TypeError(_ARRAY_LIKE_CONSTANT_MESSAGE)
