"""Core constraint metadata protocols and value aliases."""

from typing import Protocol

type ConstrainedValue = object
type UnconstrainedValue = object
type LogAbsDetJacobian = object


class Constraint(Protocol):
    """Parameter constraint metadata consumed by numerical backends."""
