"""Prior simulation domains derived from parameter constraints."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.constraints.core import Constraint
from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.core import DistributionValue


@dataclass(frozen=True)
class UnconstrainedDomain:
    """Unrestricted real-valued prior domain."""


@dataclass(frozen=True)
class ScalarIntervalDomain:
    """Scalar interval prior domain."""

    lower: DistributionValue | None
    upper: DistributionValue | None


type PriorDomain = UnconstrainedDomain | ScalarIntervalDomain


def prior_domain_for_constraint(constraint: Constraint | None) -> PriorDomain:
    """Return the constrained-value prior domain implied by a constraint."""
    if constraint is None:
        return UnconstrainedDomain()
    if isinstance(constraint, Positive):
        return ScalarIntervalDomain(lower=0.0, upper=None)
    raise TypeError(f"Unsupported prior constraint: {type(constraint).__name__}")
