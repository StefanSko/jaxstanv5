"""Integration tests for simple-normal sampling workflows."""

from __future__ import annotations

import jax.numpy as jnp
from _helpers import bind_model

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import compile_sampler, sample


@model
class SimpleNormal:
    """Scalar normal with known scale."""

    mu = Param(Normal(0, 1))
    y = Observed(Normal(mu, 1))


@model
class EmptyVectorPrior:
    """Prior-only model with a data-dependent zero-sized parameter."""

    n = Data.scalar()
    theta = Param(Normal(0, 1), size=n)


def test_simple_normal_sampling_returns_finite_samples_and_diagnostics() -> None:
    bound = bind_model(SimpleNormal, y=jnp.array(2.0))
    result = sample(bound, seed=42, num_warmup=200, num_samples=500)

    assert "mu" in result.samples
    assert result.samples["mu"].shape == (1, 500)
    assert jnp.all(jnp.isfinite(result.samples["mu"]))
    assert result.diagnostics.warmup.is_divergent.shape == (1, 200)
    assert result.diagnostics.sampling.is_divergent.shape == (1, 500)
    assert result.diagnostics.sampling.acceptance_rate.shape == (1, 500)
    assert result.diagnostics.sampling.num_integration_steps.shape == (1, 500)
    assert jnp.all(jnp.isfinite(result.diagnostics.sampling.acceptance_rate))
    assert not jnp.any(result.diagnostics.sampling.is_divergent)

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)
    assert rhat_vals["mu"] < 1.10
    assert ess_vals["mu"] > 50


def test_simple_normal_sampling_supports_multiple_chains_and_diagnostics() -> None:
    bound = bind_model(SimpleNormal, y=jnp.array(2.0))
    result = sample(bound, seed=42, num_warmup=200, num_samples=500, num_chains=4)

    assert result.samples["mu"].shape == (4, 500)
    assert jnp.all(jnp.isfinite(result.samples["mu"]))
    assert result.diagnostics.warmup.is_divergent.shape == (4, 200)
    assert result.diagnostics.sampling.is_divergent.shape == (4, 500)
    assert result.diagnostics.sampling.acceptance_rate.shape == (4, 500)
    assert result.diagnostics.sampling.num_integration_steps.shape == (4, 500)
    assert jnp.all(jnp.isfinite(result.diagnostics.sampling.acceptance_rate))
    assert not jnp.any(result.diagnostics.sampling.is_divergent)
    assert not jnp.allclose(result.samples["mu"][0], result.samples["mu"][1])

    rhat_vals = rhat(result.samples)
    ess_vals = ess(result.samples)
    assert rhat_vals["mu"] < 1.10
    assert ess_vals["mu"] > 100


def test_compiled_sampler_reuses_bound_model_for_multiple_runs() -> None:
    bound = bind_model(SimpleNormal, y=jnp.array(2.0))
    compiled = compile_sampler(bound)

    first = compiled.sample(seed=10, num_warmup=50, num_samples=100)
    second = compiled.sample(seed=11, num_warmup=50, num_samples=100)
    multi_chain = compiled.sample(seed=12, num_warmup=50, num_samples=100, num_chains=2)

    assert first.samples["mu"].shape == (1, 100)
    assert second.samples["mu"].shape == (1, 100)
    assert multi_chain.samples["mu"].shape == (2, 100)
    assert first.diagnostics.warmup.is_divergent.shape == (1, 50)
    assert first.diagnostics.sampling.is_divergent.shape == (1, 100)
    assert multi_chain.diagnostics.warmup.is_divergent.shape == (2, 50)
    assert multi_chain.diagnostics.sampling.is_divergent.shape == (2, 100)
    assert jnp.all(jnp.isfinite(first.samples["mu"]))
    assert jnp.all(jnp.isfinite(second.samples["mu"]))
    assert jnp.all(jnp.isfinite(multi_chain.samples["mu"]))
    assert not jnp.allclose(multi_chain.samples["mu"][0], multi_chain.samples["mu"][1])


def test_sampling_preserves_data_dependent_zero_sized_parameter() -> None:
    bound = bind_model(EmptyVectorPrior, n=0)

    result = sample(bound, seed=1, num_warmup=2, num_samples=3)

    assert bound.param_shapes == {"theta": (0,)}
    assert bound.n_params == 0
    assert set(result.samples) == {"theta"}
    assert result.samples["theta"].shape == (1, 3, 0)
    assert result.diagnostics.warmup.is_divergent.shape == (1, 2)
    assert result.diagnostics.sampling.is_divergent.shape == (1, 3)
    assert not jnp.any(result.diagnostics.warmup.is_divergent)
    assert not jnp.any(result.diagnostics.sampling.is_divergent)
    assert rhat(result.samples) == {}
    assert ess(result.samples) == {}
