"""Truncated distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.distributions.core import Distribution, DistributionParameter


@dataclass(frozen=True)
class Truncated:
    """Distribution restricted to explicit lower and/or upper scalar bounds."""

    base: Distribution
    lower: DistributionParameter | None = None
    upper: DistributionParameter | None = None

    def __post_init__(self) -> None:
        """Require at least one truncation bound."""
        if self.lower is None and self.upper is None:
            raise ValueError("Truncated distributions require at least one bound")
