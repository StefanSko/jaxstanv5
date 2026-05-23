"""Slice test — full inference pipeline: model → bind → sample → diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from jaxstanv5 import Observed, Param, model
from jaxstanv5.diagnostics.core import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference.core import sample


@model
class SimpleNormal:
    """Scalar normal with known scale."""

    mu = Param(Normal(0, 1))
    y = Observed(Normal(mu, 1))


def test_sample_simple_model() -> None:
    bound = SimpleNormal.bind(y=jnp.array(2.0))
    result = sample(bound, seed=42, num_warmup=200, num_samples=500)

    # Shape: (1 chain, 500 draws) per scalar param
    assert "mu" in result.samples
    assert result.samples["mu"].shape == (1, 500)
    assert jnp.all(jnp.isfinite(result.samples["mu"]))

    # Diagnostics
    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)
    assert rhat_vals["mu"] < 1.10
    assert ess_vals["mu"] > 50
