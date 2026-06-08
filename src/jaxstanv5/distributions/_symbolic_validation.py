"""Validation helpers for symbolic values hidden inside opaque distributions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
    return _slotted_object_attributes(value)


def _slotted_object_attributes(value: object) -> dict[str, object] | None:
    slot_names = _object_slot_names(value)
    if not slot_names:
        return None

    attributes: dict[str, object] = {}
    for name in slot_names:
        try:
            attributes[name] = getattr(value, name)
        except AttributeError:
            continue
    return attributes


def _object_slot_names(value: object) -> tuple[str, ...]:
    names: list[str] = []
    for cls in type(value).__mro__:
        raw_slots = getattr(cls, "__slots__", ())
        if isinstance(raw_slots, str):
            _append_slot_name(names, _slot_attribute_name(cls, raw_slots))
        elif isinstance(raw_slots, Iterable):
            for raw_name in raw_slots:
                if isinstance(raw_name, str):
                    _append_slot_name(names, _slot_attribute_name(cls, raw_name))
    return tuple(names)


def _slot_attribute_name(cls: type[object], name: str) -> str:
    if name.startswith("__") and not name.endswith("__"):
        class_name = cls.__name__.lstrip("_")
        if class_name:
            return f"_{class_name}{name}"
    return name


def _append_slot_name(names: list[str], name: str) -> None:
    if name in {"__dict__", "__weakref__"}:
        return
    if name not in names:
        names.append(name)
