"""Tests for prior and prior-predictive simulation."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from jaxstanv5 import Data, Observed, Param, PartiallyObserved, model
from jaxstanv5.constraints import Ordered, Positive
from jaxstanv5.distributions import MultivariateNormal, Normal, OrderedLogistic, Uniform
from jaxstanv5.distributions.core import DistributionValue, LogProbability
from jaxstanv5.simulation import simulate_prior_predictive


@model
class PriorOnlyNormal:
    """Prior-only scalar model."""

    mu = Param(Normal(0.0, 1.0))


@model
class NormalWithObserved:
    """Scalar prior with vector observed predictive draws."""

    mu = Param(Normal(0.0, 1.0))
    y = Observed(Normal(mu, 1.0))


@model
class PositiveScalePrior:
    """Positive-constrained prior-only model."""

    sigma = Param(Normal(0.0, 1.0), constraint=Positive())


@model
class PositiveUniformPrior:
    """Positive-constrained Uniform prior-only model."""

    theta = Param(Uniform(-1.0, 2.0), constraint=Positive())


class UnsupportedDistribution:
    """Distribution with no simulation support."""

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return jnp.zeros_like(jnp.asarray(x))


@model
class DataSizedPrior:
    """Data-dependent vector prior."""

    n = Data.scalar()
    theta = Param(Normal(0.0, 1.0), size=n)


@model
class MultivariateNormalPrior:
    """Event-shaped multivariate Normal prior."""

    n = Data.scalar()
    chol = Data.matrix(n, n)
    f = Param(MultivariateNormal(0.0, chol), size=n)


@model
class BatchedNormalPrior:
    """Scalar-event prior with distribution batch shape matching parameter shape."""

    loc = Data.vector(3)
    theta = Param(Normal(loc, 1.0), size=3)


@model
class MultivariateNormalObserved:
    """Event-shaped multivariate Normal prior and observed site."""

    n = Data.scalar()
    chol = Data.matrix(n, n)
    mu = Param(MultivariateNormal(0.0, chol), size=n)
    y = Observed(MultivariateNormal(mu, chol))


@model
class FixedKernelGpPriorPredictive:
    """Fixed-kernel GP prior predictive model."""

    n = Data.scalar()
    chol = Data.matrix(n, n)
    obs_sd = Data.scalar()
    f = Param(MultivariateNormal(0.0, chol), size=n)
    y = Observed(Normal(f, obs_sd))


@model
class OrderedOrdinalPriorPredictive:
    """Ordinal prior predictive model with ordered cutpoints."""

    n_cutpoints = Data.scalar()
    x = Data.vector()
    beta = Param(Normal(0.0, 1.0))
    cutpoints = Param(Normal(0.0, 2.0), size=n_cutpoints, constraint=Ordered())
    eta = beta * x
    y = Observed(OrderedLogistic(eta, cutpoints))


@model
class UnsupportedPrior:
    """Prior using an unsupported distribution."""

    theta = Param(UnsupportedDistribution())


@model
class PartialObservedPriorPredictive:
    """Partially observed vectors are not prior-predictive simulation nodes in v1."""

    n = Data.scalar()
    n_obs = Data.scalar()
    n_mis = Data.scalar()
    chol = Data.matrix(n, n)
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector(n_mis)
    observed_values = Data.vector(n_obs)
    y = PartiallyObserved.vector(
        MultivariateNormal(0.0, chol),
        length=n,
        observed=observed_values,
        observed_idx=observed_idx,
        missing_idx=missing_idx,
    )


def test_simulate_prior_predictive_draws_prior_only_parameters_with_vmap_shape() -> None:
    result = simulate_prior_predictive(PriorOnlyNormal, seed=1, num_samples=7)

    assert result.parameters["mu"].shape == (7,)
    assert result.observed == {}
    assert result.data == {}
    assert jnp.all(jnp.isfinite(result.parameters["mu"]))


def test_simulate_prior_predictive_is_deterministic_for_seed() -> None:
    first = simulate_prior_predictive(PriorOnlyNormal, seed=101, num_samples=4)
    second = simulate_prior_predictive(PriorOnlyNormal, seed=101, num_samples=4)

    assert jnp.allclose(first.parameters["mu"], second.parameters["mu"])


def test_simulate_prior_predictive_draws_observed_values_with_requested_shape() -> None:
    result = simulate_prior_predictive(
        NormalWithObserved,
        seed=2,
        num_samples=5,
        observed_shapes={"y": (3,)},
    )

    assert result.parameters["mu"].shape == (5,)
    assert result.observed["y"].shape == (5, 3)
    assert jnp.all(jnp.isfinite(result.observed["y"]))


def test_simulate_prior_predictive_draws_positive_constrained_parameters() -> None:
    result = simulate_prior_predictive(PositiveScalePrior, seed=3, num_samples=9)

    assert result.parameters["sigma"].shape == (9,)
    assert jnp.all(result.parameters["sigma"] > 0.0)


def test_simulate_prior_predictive_draws_positive_constrained_uniform_parameters() -> None:
    result = simulate_prior_predictive(PositiveUniformPrior, seed=31, num_samples=20)

    assert result.parameters["theta"].shape == (20,)
    assert jnp.all(result.parameters["theta"] >= 0.0)
    assert jnp.all(result.parameters["theta"] <= 2.0)


def test_simulate_prior_predictive_resolves_data_dependent_parameter_shape() -> None:
    result = simulate_prior_predictive(DataSizedPrior, seed=4, num_samples=6, data={"n": 4})

    assert result.parameters["theta"].shape == (6, 4)
    assert result.data["n"].shape == ()


def test_simulate_prior_predictive_draws_multivariate_normal_prior_event() -> None:
    result = simulate_prior_predictive(
        MultivariateNormalPrior,
        seed=41,
        num_samples=5,
        data={"n": 3, "chol": jnp.eye(3)},
    )

    assert result.parameters["f"].shape == (5, 3)
    assert jnp.all(jnp.isfinite(result.parameters["f"]))


def test_simulate_prior_predictive_does_not_double_distribution_batch_shape() -> None:
    result = simulate_prior_predictive(
        BatchedNormalPrior,
        seed=42,
        num_samples=5,
        data={"loc": jnp.array([0.0, 1.0, 2.0])},
    )

    assert result.parameters["theta"].shape == (5, 3)


def test_simulate_prior_predictive_draws_multivariate_normal_observed_event() -> None:
    result = simulate_prior_predictive(
        MultivariateNormalObserved,
        seed=43,
        num_samples=5,
        data={"n": 3, "chol": jnp.eye(3)},
    )

    assert result.parameters["mu"].shape == (5, 3)
    assert result.observed["y"].shape == (5, 3)


def test_simulate_prior_predictive_draws_ordered_cutpoints_and_ordinal_observations() -> None:
    result = simulate_prior_predictive(
        OrderedOrdinalPriorPredictive,
        seed=45,
        num_samples=7,
        data={"n_cutpoints": 2, "x": jnp.linspace(-1.0, 1.0, 10)},
    )

    cutpoints = result.parameters["cutpoints"]
    y = result.observed["y"]
    assert cutpoints.shape == (7, 2)
    assert jnp.all(cutpoints[:, 1] > cutpoints[:, 0])
    assert y.shape == (7, 10)
    assert jnp.all(y >= 0)
    assert jnp.all(y <= 2)


def test_simulate_prior_predictive_draws_fixed_kernel_gp_shapes() -> None:
    result = simulate_prior_predictive(
        FixedKernelGpPriorPredictive,
        seed=44,
        num_samples=5,
        data={"n": 3, "chol": jnp.eye(3), "obs_sd": 0.2},
    )

    assert result.parameters["f"].shape == (5, 3)
    assert result.observed["y"].shape == (5, 3)


def test_simulate_prior_predictive_rejects_non_lower_triangular_mvn_scale_tril_data() -> None:
    with pytest.raises(ValueError, match="jnp.linalg.cholesky"):
        simulate_prior_predictive(
            FixedKernelGpPriorPredictive,
            seed=44,
            num_samples=5,
            data={
                "n": 3,
                "chol": jnp.asarray([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "obs_sd": 0.2,
            },
        )


def test_simulate_prior_predictive_rejects_wrong_shaped_data() -> None:
    with pytest.raises(ValueError, match="Data 'chol' has wrong shape"):
        simulate_prior_predictive(
            FixedKernelGpPriorPredictive,
            seed=44,
            num_samples=5,
            data={"n": 3, "chol": jnp.eye(2), "obs_sd": 0.2},
        )


def test_simulate_prior_predictive_rejects_partially_observed_models() -> None:
    with pytest.raises(TypeError, match="PartiallyObserved declarations are not supported"):
        simulate_prior_predictive(
            PartialObservedPriorPredictive,
            seed=46,
            num_samples=2,
            data={
                "n": 3,
                "n_obs": 2,
                "n_mis": 1,
                "chol": jnp.eye(3),
                "observed_idx": jnp.asarray([0, 2]),
                "missing_idx": jnp.asarray([1]),
                "observed_values": jnp.asarray([1.0, -1.0]),
            },
        )


def test_simulate_prior_predictive_rejects_missing_data() -> None:
    with pytest.raises(ValueError, match="Missing model data"):
        simulate_prior_predictive(DataSizedPrior, seed=5, num_samples=2)


def test_simulate_prior_predictive_rejects_extra_data() -> None:
    with pytest.raises(ValueError, match="Unexpected model data"):
        simulate_prior_predictive(PriorOnlyNormal, seed=5, num_samples=2, data={"x": 1.0})


def test_simulate_prior_predictive_rejects_unknown_observed_shape() -> None:
    with pytest.raises(ValueError, match="Unexpected observed shapes"):
        simulate_prior_predictive(
            PriorOnlyNormal,
            seed=6,
            num_samples=2,
            observed_shapes={"y": (1,)},
        )


def test_simulate_prior_predictive_rejects_negative_observed_shape() -> None:
    with pytest.raises(ValueError, match="Observed shape dimensions must be non-negative"):
        simulate_prior_predictive(
            NormalWithObserved,
            seed=7,
            num_samples=2,
            observed_shapes={"y": (-1,)},
        )


def test_simulate_prior_predictive_rejects_unsupported_prior_distribution() -> None:
    with pytest.raises(TypeError, match="Unsupported prior distribution"):
        simulate_prior_predictive(UnsupportedPrior, seed=8, num_samples=2)


def test_simulate_prior_predictive_rejects_undecorated_model() -> None:
    class Undecorated:
        theta = Param(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="decorated with @model"):
        simulate_prior_predictive(Undecorated, seed=9, num_samples=2)


def test_simulate_prior_predictive_rejects_non_positive_num_samples() -> None:
    with pytest.raises(ValueError, match="num_samples must be at least 1"):
        simulate_prior_predictive(PriorOnlyNormal, seed=10, num_samples=0)
