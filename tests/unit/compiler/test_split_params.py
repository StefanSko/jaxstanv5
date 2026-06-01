"""Unit tests for compiler parameter vector splitting."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5.compiler.core import _split_params


def test_split_params_zero_sized_parameter_does_not_advance_offset() -> None:
    """Zero-sized parameters consume no flat-vector entries."""
    flat = jnp.array([10.0, 20.0])
    shapes: dict[str, tuple[int, ...]] = {"empty": (0,), "first": (), "second": ()}

    result = _split_params(flat, shapes)

    assert result["empty"].shape == (0,)
    assert jnp.allclose(result["first"], jnp.array(10.0))
    assert jnp.allclose(result["second"], jnp.array(20.0))


def test_split_params_rejects_non_vector_input() -> None:
    flat = jnp.array([[1.0, 2.0]])
    shapes: dict[str, tuple[int, ...]] = {"theta": (2,)}

    with pytest.raises(ValueError, match="one-dimensional"):
        _split_params(flat, shapes)


def test_split_params_rejects_wrong_length_input() -> None:
    shapes: dict[str, tuple[int, ...]] = {"alpha": (), "beta": (2,)}

    with pytest.raises(ValueError, match="expected 3, got 2"):
        _split_params(jnp.array([1.0, 2.0]), shapes)

    with pytest.raises(ValueError, match="expected 3, got 4"):
        _split_params(jnp.array([1.0, 2.0, 3.0, 4.0]), shapes)
