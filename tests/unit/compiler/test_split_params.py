"""Unit tests for compiler parameter vector splitting."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5.compiler.core import split_params


def test_split_params_zero_sized_parameter_does_not_advance_offset() -> None:
    """Zero-sized parameters consume no flat-vector entries."""
    flat = jnp.array([10.0, 20.0])
    shapes: dict[str, tuple[int, ...]] = {"empty": (0,), "first": (), "second": ()}

    result = split_params(flat, shapes)

    assert result["empty"].shape == (0,)
    assert jnp.allclose(result["first"], jnp.array(10.0))
    assert jnp.allclose(result["second"], jnp.array(20.0))
