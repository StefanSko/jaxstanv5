"""Discrete count distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.distributions.core import DiscreteDistribution, DistributionParameter


@dataclass(frozen=True)
class Bernoulli(DiscreteDistribution):
    """Bernoulli distribution metadata parameterized by success probability."""

    probs: DistributionParameter


@dataclass(frozen=True)
class Poisson(DiscreteDistribution):
    """Poisson distribution metadata parameterized by positive rate."""

    rate: DistributionParameter


@dataclass(frozen=True)
class Binomial(DiscreteDistribution):
    """Binomial distribution metadata parameterized by count and probability."""

    total_count: DistributionParameter
    probs: DistributionParameter


@dataclass(frozen=True)
class BetaBinomial(DiscreteDistribution):
    """Beta-binomial distribution metadata parameterized by count and concentrations."""

    total_count: DistributionParameter
    alpha: DistributionParameter
    beta: DistributionParameter


@dataclass(frozen=True)
class NegativeBinomial(DiscreteDistribution):
    """Negative-binomial NB2 distribution metadata."""

    mean: DistributionParameter
    overdispersion: DistributionParameter
