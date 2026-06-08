"""Validation helpers for symbolic values hidden inside opaque distributions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass

from jaxstanv5.distributions.core import SymbolicDistributionParameter


def reject_opaque_symbolic_distribution(value: object) -> None:
    """Reject opaque non-dataclass values that contain symbolic parameters."""
    if not contains_symbolic_distribution_parameter(value):
        return

    raise TypeError(
        "Custom distributions with symbolic parameters must be dataclasses. "
        f"{type(value).__name__} is opaque to jaxstanv5 because it is not a dataclass, "
        "so symbolic fields cannot be resolved. Add @dataclass(frozen=True) to the "
        "distribution class or pass only concrete distribution parameters."
    )


def contains_symbolic_distribution_parameter(value: object) -> bool:
    """Return whether ``value`` contains declaration-time symbolic parameters."""
    return _contains_symbolic_distribution_parameter(value, frozenset())


def _contains_symbolic_distribution_parameter(value: object, seen: frozenset[int]) -> bool:
    if isinstance(value, SymbolicDistributionParameter):
        return True
    if isinstance(value, str | bytes):
        return False

    value_id = id(value)
    if value_id in seen:
        return False
    next_seen = seen | {value_id}

    if is_dataclass(value) and not isinstance(value, type):
        return any(
            _contains_symbolic_distribution_parameter(getattr(value, field.name), next_seen)
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return any(
            _contains_symbolic_distribution_parameter(key, next_seen)
            or _contains_symbolic_distribution_parameter(item, next_seen)
            for key, item in value.items()
        )
    if isinstance(value, Sequence):
        return any(_contains_symbolic_distribution_parameter(item, next_seen) for item in value)

    attributes = _object_attributes(value)
    if attributes is None:
        return False
    return any(
        _contains_symbolic_distribution_parameter(item, next_seen) for item in attributes.values()
    )


def _object_attributes(value: object) -> Mapping[str, object] | None:
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    return None
