"""Normal distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp

from jaxstanv5.distributions.core import (
    DistributionParameter,
    DistributionValue,
    LogProbability,
    concrete_parameter,
)


@dataclass(frozen=True)
class Normal:
    """Univariate normal distribution supporting JAX broadcasting."""

    loc: DistributionParameter
    scale: DistributionParameter

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise normal log-density for ``x``."""
        loc = jnp.asarray(concrete_parameter(self.loc))
        scale = jnp.asarray(concrete_parameter(self.scale))
        value = jnp.asarray(x)
        standardized = (value - loc) / scale
        return -0.5 * standardized**2 - jnp.log(scale) - 0.5 * math.log(2.0 * math.pi)
