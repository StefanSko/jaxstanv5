"""North-star tests for partially observed continuous random vectors."""

from __future__ import annotations

import jax.numpy as jnp
from _reference_models import partially_observed_mvn_fixture

from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.distributions import MultivariateNormal
from jaxstanv5.inference import sample


def test_partially_observed_mvn_log_density_matches_joint_density() -> None:
    fixture = partially_observed_mvn_fixture()
    log_density = compile_log_density(fixture.bound)
    distribution = MultivariateNormal(0.0, fixture.chol)

    for missing_value in (-1.25, 0.0, 1.5):
        q = jnp.asarray([missing_value])
        full_y = jnp.zeros((3,))
        full_y = full_y.at[fixture.observed_idx].set(fixture.observed_values)
        full_y = full_y.at[fixture.missing_idx].set(q)

        actual = log_density(q)
        expected = distribution.log_prob(full_y)

        assert jnp.allclose(actual, expected, atol=1e-6)


def test_partially_observed_mvn_samples_match_analytic_conditional() -> None:
    fixture = partially_observed_mvn_fixture()

    result = sample(
        fixture.bound,
        seed=202,
        num_warmup=400,
        num_samples=800,
        num_chains=4,
        target_acceptance_rate=0.9,
    )

    draws = result.samples["y"][..., 0].reshape((-1,))
    sample_mean = float(jnp.mean(draws))
    sample_variance = float(jnp.var(draws))

    assert abs(sample_mean - fixture.conditional_mean) < 0.08
    assert abs(sample_variance - fixture.conditional_variance) < 0.12
    assert int(jnp.sum(result.diagnostics.sampling.is_divergent)) == 0
