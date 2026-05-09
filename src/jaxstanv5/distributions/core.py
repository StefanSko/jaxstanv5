"""Core distribution types."""

from typing import Protocol

import jax
from jax.typing import ArrayLike

type DistributionValue = ArrayLike
type DistributionParameter = ArrayLike
type LogProbability = jax.Array


class Distribution(Protocol):
    """Probability distribution with an element-wise log-density."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        """Return element-wise log-probability for ``x``."""
        ...
