"""Ordered-vector constraint metadata."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Ordered:
    """Constraint metadata for strictly increasing vectors along the last axis."""
