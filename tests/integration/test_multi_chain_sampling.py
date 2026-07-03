"""Integration tests for public multi-chain sampling semantics."""

from __future__ import annotations

import jax.numpy as jnp
from _helpers import bind_model
from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated

from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.inference import sample


@model
class SimpleNormalForSeedChecks:
    """Scalar normal model for deterministic seeding checks."""

    mu = Param(Normal(0, 1))
    y = Observed(Normal(mu, 1))


@model
class VectorAndConstrainedNormal:
    """Model with vector and constrained scalar parameters."""

    n = Data.scalar()
    theta = Param(Normal(0, 1), size=n)
    sigma = Param(Truncated(Normal(0, 1), lower=0.0), constraint=Positive())
    y = Observed(Normal(theta, 1))


def test_multi_chain_sampling_is_reproducible_for_same_seed() -> None:
    bound = bind_model(SimpleNormalForSeedChecks, y=jnp.array(1.0))

    first = sample(bound, seed=123, num_warmup=80, num_samples=100, num_chains=3)
    second = sample(bound, seed=123, num_warmup=80, num_samples=100, num_chains=3)
    different_seed = sample(bound, seed=124, num_warmup=80, num_samples=100, num_chains=3)

    assert jnp.allclose(first.samples["mu"], second.samples["mu"])
    assert not jnp.allclose(first.samples["mu"], different_seed.samples["mu"])
    assert not jnp.allclose(first.samples["mu"][0], first.samples["mu"][1])


def test_multi_chain_sampling_preserves_shapes_constraints_and_diagnostics() -> None:
    y_data = jnp.array([-0.5, 0.0, 0.5])
    bound = bind_model(VectorAndConstrainedNormal, n=3, y=y_data)

    result = sample(bound, seed=321, num_warmup=200, num_samples=300, num_chains=3)

    assert result.samples["theta"].shape == (3, 300, 3)
    assert result.samples["sigma"].shape == (3, 300)
    assert result.diagnostics.warmup.is_divergent.shape == (3, 200)
    assert result.diagnostics.sampling.is_divergent.shape == (3, 300)
    assert result.diagnostics.sampling.acceptance_rate.shape == (3, 300)
    assert result.diagnostics.sampling.num_integration_steps.shape == (3, 300)
    assert jnp.all(jnp.isfinite(result.samples["theta"]))
    assert jnp.all(jnp.isfinite(result.samples["sigma"]))
    assert jnp.all(result.samples["sigma"] > 0.0)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)

    assert rhat_vals["theta"] < 1.25
    assert rhat_vals["sigma"] < 1.25
    assert ess_vals["theta"] > 10
    assert ess_vals["sigma"] > 10
