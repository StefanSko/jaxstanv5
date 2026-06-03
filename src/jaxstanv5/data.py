"""Small data-preparation helpers for explicit model inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PartialVector:
    """Explicit partition of a rank-1 vector into observed and missing entries."""

    length: int
    n_observed: int
    n_missing: int
    observed_idx: jax.Array
    missing_idx: jax.Array
    observed_values: jax.Array

    @classmethod
    def from_nan(cls, values: object) -> Self:
        """Build an explicit partial-vector partition from NaN sentinels.

        This helper is intentionally outside the model declaration language: NaNs
        are a data-preparation convenience, not model semantics.
        """
        vector = jnp.asarray(values)
        if vector.ndim != 1:
            raise ValueError("PartialVector.from_nan requires a rank-1 vector")

        if jnp.issubdtype(vector.dtype, jnp.inexact):
            missing_mask = jnp.isnan(vector)
        else:
            missing_mask = jnp.zeros(vector.shape, dtype=bool)

        missing_flags = tuple(bool(flag) for flag in missing_mask.tolist())
        observed_positions = tuple(
            index for index, missing in enumerate(missing_flags) if not missing
        )
        missing_positions = tuple(index for index, missing in enumerate(missing_flags) if missing)

        observed_idx = jnp.asarray(observed_positions, dtype=jnp.int32)
        missing_idx = jnp.asarray(missing_positions, dtype=jnp.int32)
        observed_values = vector[observed_idx]

        return cls(
            length=vector.shape[0],
            n_observed=len(observed_positions),
            n_missing=len(missing_positions),
            observed_idx=observed_idx,
            missing_idx=missing_idx,
            observed_values=observed_values,
        )
