"""End-to-end IR serialization over the reference model corpus."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from _reference_models import (
    beta_binomial_logistic_fixture,
    beta_logistic_fixture,
    binomial_logistic_fixture,
    eight_schools_fixture,
    exponential_rate_fixture,
    fixed_kernel_gp_fixture,
    hierarchical_beta_binomial_logistic_varying_slopes_fixture,
    hierarchical_beta_regression_logistic_varying_slopes_fixture,
    hierarchical_binomial_logistic_varying_slopes_fixture,
    hierarchical_negative_binomial_log_rate_varying_slopes_fixture,
    hierarchical_poisson_varying_slopes_fixture,
    multivariate_normal_likelihood_fixture,
    negative_binomial_log_rate_fixture,
    non_centered_known_scale_fixture,
    ordinal_logistic_regression_fixture,
    partially_observed_mvn_fixture,
    poisson_log_rate_fixture,
    robust_regression_fixture,
    student_t_location_fixture,
)

from jaxstanv5.compiler import compile_log_density
from jaxstanv5.ir import bindable_from_meta, canonical_bytes, meta_from_dict, meta_to_dict
from jaxstanv5.model.bound import BoundModel


class _BoundFixture(Protocol):
    bound: BoundModel


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


type FixtureBuilder = Callable[[], _BoundFixture]

CORPUS: tuple[FixtureBuilder, ...] = (
    eight_schools_fixture,
    exponential_rate_fixture,
    poisson_log_rate_fixture,
    binomial_logistic_fixture,
    beta_binomial_logistic_fixture,
    beta_logistic_fixture,
    negative_binomial_log_rate_fixture,
    robust_regression_fixture,
    hierarchical_poisson_varying_slopes_fixture,
    hierarchical_binomial_logistic_varying_slopes_fixture,
    hierarchical_beta_binomial_logistic_varying_slopes_fixture,
    hierarchical_beta_regression_logistic_varying_slopes_fixture,
    hierarchical_negative_binomial_log_rate_varying_slopes_fixture,
    multivariate_normal_likelihood_fixture,
    partially_observed_mvn_fixture,
    ordinal_logistic_regression_fixture,
    fixed_kernel_gp_fixture,
    non_centered_known_scale_fixture,
    student_t_location_fixture,
)

EQUIVALENCE_CASES: tuple[FixtureBuilder, ...] = (
    eight_schools_fixture,
    partially_observed_mvn_fixture,
    ordinal_logistic_regression_fixture,
)


def _round_trip_meta(bound: BoundModel) -> BoundModel:
    """Serialize, decode, and rebind through the public IR path."""
    document = json.loads(canonical_bytes(bound.meta).decode("utf-8"))
    rebuilt = cast(_BindableModel, bindable_from_meta(meta_from_dict(document)))
    return rebuilt.bind(**bound.data)


@pytest.mark.parametrize("build_fixture", CORPUS, ids=lambda builder: builder.__name__)
def test_corpus_metadata_round_trips_exactly(build_fixture: FixtureBuilder) -> None:
    bound = build_fixture().bound

    decoded = meta_from_dict(json.loads(json.dumps(meta_to_dict(bound.meta))))

    assert decoded == bound.meta
    assert canonical_bytes(decoded) == canonical_bytes(bound.meta)


@pytest.mark.parametrize("build_fixture", EQUIVALENCE_CASES, ids=lambda builder: builder.__name__)
def test_round_tripped_model_compiles_identical_log_density(
    build_fixture: FixtureBuilder,
) -> None:
    bound = build_fixture().bound

    rebound = _round_trip_meta(bound)

    assert rebound.param_shapes == bound.param_shapes
    assert rebound.n_params == bound.n_params

    original = compile_log_density(bound)
    rebuilt = compile_log_density(rebound)
    for seed in range(3):
        q = 0.1 * jax.random.normal(jax.random.PRNGKey(seed), (bound.n_params,))
        assert float(original(q)) == float(rebuilt(q))
        assert jnp.array_equal(jax.grad(original)(q), jax.grad(rebuilt)(q))
