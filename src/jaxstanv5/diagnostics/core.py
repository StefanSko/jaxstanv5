"""Core diagnostics — convergence and efficiency checks for MCMC samples."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from blackjax.diagnostics import effective_sample_size, potential_scale_reduction

_MIN_DRAWS_FOR_SPLIT_DIAGNOSTICS = 4


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


def _validate_diagnostic_sample_array(name: str, arr: jax.Array) -> None:
    """Validate the chain/draw axes needed by split-chain diagnostics."""
    if arr.ndim < 2:
        raise ValueError(
            f"Diagnostic samples for parameter {name!r} must have shape (chain, draw, *param_shape)"
        )
    num_draws = arr.shape[1]
    if num_draws < _MIN_DRAWS_FOR_SPLIT_DIAGNOSTICS:
        raise ValueError(
            f"Diagnostics for parameter {name!r} require at least "
            f"{_MIN_DRAWS_FOR_SPLIT_DIAGNOSTICS} post-warmup draws per chain; "
            f"got {num_draws}"
        )


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
    result: dict[str, float] = {}
    for name, arr in samples.items():
        if not _has_sample_values(arr):
            continue
        _validate_diagnostic_sample_array(name, arr)
        result[name] = float(jnp.max(potential_scale_reduction(_split_chains(arr))))
    return result


def ess(samples: dict[str, jax.Array]) -> dict[str, float]:
    """Compute split effective sample size for each non-empty parameter.

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
    result: dict[str, float] = {}
    for name, arr in samples.items():
        if not _has_sample_values(arr):
            continue
        _validate_diagnostic_sample_array(name, arr)
        result[name] = float(jnp.min(effective_sample_size(_split_chains(arr))))
    return result
