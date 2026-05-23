"""Core diagnostics — convergence and efficiency checks for MCMC samples."""

from __future__ import annotations

import jax
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction


def rhat(samples: dict[str, jax.Array]) -> dict[str, float]:
    """Compute split R-hat for each parameter.

    Parameters
    ----------
    samples : dict[str, jax.Array]
        Parameter names to arrays of shape ``(chain, draw)``.

    Returns
    -------
    dict[str, float]
        R-hat value per parameter.  Values near 1 indicate convergence.
    """
    return {name: float(potential_scale_reduction(arr)) for name, arr in samples.items()}


def ess(samples: dict[str, jax.Array]) -> dict[str, float]:
    """Compute effective sample size for each parameter.

    Parameters
    ----------
    samples : dict[str, jax.Array]
        Parameter names to arrays of shape ``(chain, draw)``.

    Returns
    -------
    dict[str, float]
        ESS per parameter.  Higher is better.
    """
    return {name: float(effective_sample_size(arr)) for name, arr in samples.items()}
