from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5.data import PartialVector


def test_partial_vector_from_nan_extracts_explicit_partition() -> None:
    raw = jnp.asarray([1.2, jnp.nan, -0.3, jnp.nan, 0.8])

    partial = PartialVector.from_nan(raw)

    assert partial.length == 5
    assert partial.n_observed == 3
    assert partial.n_missing == 2
    assert jnp.array_equal(partial.observed_idx, jnp.asarray([0, 2, 4], dtype=jnp.int32))
    assert jnp.array_equal(partial.missing_idx, jnp.asarray([1, 3], dtype=jnp.int32))
    assert jnp.allclose(partial.observed_values, jnp.asarray([1.2, -0.3, 0.8]))


def test_partial_vector_from_nan_accepts_all_observed_vector() -> None:
    partial = PartialVector.from_nan(jnp.asarray([1.0, 2.0, 3.0]))

    assert partial.length == 3
    assert partial.n_observed == 3
    assert partial.n_missing == 0
    assert jnp.array_equal(partial.observed_idx, jnp.asarray([0, 1, 2], dtype=jnp.int32))
    assert partial.missing_idx.shape == (0,)
    assert jnp.allclose(partial.observed_values, jnp.asarray([1.0, 2.0, 3.0]))


def test_partial_vector_from_nan_accepts_all_missing_vector() -> None:
    partial = PartialVector.from_nan(jnp.asarray([jnp.nan, jnp.nan]))

    assert partial.length == 2
    assert partial.n_observed == 0
    assert partial.n_missing == 2
    assert partial.observed_idx.shape == (0,)
    assert jnp.array_equal(partial.missing_idx, jnp.asarray([0, 1], dtype=jnp.int32))
    assert partial.observed_values.shape == (0,)


def test_partial_vector_from_nan_rejects_non_vector_input() -> None:
    with pytest.raises(ValueError, match="PartialVector.from_nan requires a rank-1 vector"):
        PartialVector.from_nan(jnp.ones((2, 2)))
