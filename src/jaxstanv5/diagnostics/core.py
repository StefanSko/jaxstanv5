"""Core diagnostics — convergence and efficiency checks for MCMC samples."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction


def _split_chains(arr: jax.Array) -> jax.Array:
    """Split every chain in half along the draw axis, dropping an odd final draw."""
    n_draws = arr.shape[1]
    half = n_draws // 2
    first = arr[:, :half]
    second = arr[:, half : 2 * half]
    return jnp.concatenate([first, second], axis=0)


def _has_sample_values(arr: jax.Array) -> bool:
    """Return whether a sample array has at least one scalar coordinate."""
    return arr.size > 0


def rhat(samples: dict[str, jax.Array]) -> dict[str, float]:
    """Compute split R-hat for each non-empty parameter.

    Parameters
    ----------
    samples : dict[str, jax.Array]
        Parameter names to arrays of shape ``(chain, draw, *param_shape)``.

    Returns
    -------
    dict[str, float]
        Maximum R-hat per parameter (conservative for vector params). Zero-sized
        parameter arrays are omitted because they have no scalar coordinates.
    """
    return {
        name: float(jnp.max(potential_scale_reduction(_split_chains(arr))))
        for name, arr in samples.items()
        if _has_sample_values(arr)
    }


def ess(samples: dict[str, jax.Array]) -> dict[str, float]:
    """Compute effective sample size for each non-empty parameter.

    Parameters
    ----------
    samples : dict[str, jax.Array]
        Parameter names to arrays of shape ``(chain, draw, *param_shape)``.

    Returns
    -------
    dict[str, float]
        Minimum ESS per parameter (conservative for vector params). Zero-sized
        parameter arrays are omitted because they have no scalar coordinates.
    """
    return {
        name: float(jnp.min(effective_sample_size(arr)))
        for name, arr in samples.items()
        if _has_sample_values(arr)
    }
