"""Scalar continuous distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.distributions.core import DistributionParameter


@dataclass(frozen=True)
class Normal:
    """Univariate normal distribution metadata."""

    loc: DistributionParameter
    scale: DistributionParameter


@dataclass(frozen=True)
class HalfNormal:
    """Half-Normal distribution metadata on non-negative real values."""

    scale: DistributionParameter


@dataclass(frozen=True)
class StudentT:
    """Student-t distribution metadata parameterized by df, location, and scale."""

    df: DistributionParameter
    loc: DistributionParameter
    scale: DistributionParameter


@dataclass(frozen=True)
class Exponential:
    """Exponential distribution metadata parameterized by positive rate."""

    rate: DistributionParameter


@dataclass(frozen=True)
class Uniform:
    """Continuous uniform distribution metadata."""

    low: DistributionParameter
    high: DistributionParameter


@dataclass(frozen=True)
class Beta:
    """Beta distribution metadata parameterized by positive concentrations."""

    alpha: DistributionParameter
    beta: DistributionParameter
