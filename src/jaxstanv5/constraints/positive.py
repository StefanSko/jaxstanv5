"""Positive constraint metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Positive:
    """Constraint metadata for strictly positive values."""
