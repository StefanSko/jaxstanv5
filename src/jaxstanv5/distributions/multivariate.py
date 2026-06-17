"""Multivariate distribution metadata."""

from __future__ import annotations

from dataclasses import dataclass

from jaxstanv5.distributions.core import DistributionParameter


@dataclass(frozen=True)
class MultivariateNormal:
    """Event-wise multivariate Normal metadata with a Cholesky scale factor."""

    mean: DistributionParameter
    scale_tril: DistributionParameter
