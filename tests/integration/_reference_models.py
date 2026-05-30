"""Lazy shared model fixtures for distribution-coverage validation.

The functions in this module intentionally import target distributions inside the
fixture builders. Missing distribution support should fail the specific test that
needs it, not module import or pytest collection.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxstanv5.model.bound import BoundModel

try:  # pytest imports integration helpers as siblings; scripts import through ``integration``.
    from _helpers import bind_model
except ModuleNotFoundError:  # pragma: no cover - exercised by out-of-band scripts.
    from integration._helpers import bind_model


@dataclass(frozen=True)
class EightSchoolsFixture:
    """Bound non-centered eight-schools model and its observed data."""

    bound: BoundModel
    y: jax.Array
    sigma: jax.Array


@dataclass(frozen=True)
class ExponentialRateFixture:
    """Bound Exponential-rate model and data needed for references."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    prior_scale: float
    true_rate: float


@dataclass(frozen=True)
class RobustRegressionFixture:
    """Bound robust linear regression with Student-t likelihood."""

    bound: BoundModel
    x: jax.Array
    y: jax.Array
    nu: float


@dataclass(frozen=True)
class MultivariateNormalLikelihoodFixture:
    """Bound single-observation multivariate-normal likelihood model."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    prior_mean: jax.Array
    prior_scale: float
    covariance: jax.Array
    chol: jax.Array


@dataclass(frozen=True)
class FixedKernelGpFixture:
    """Bound fixed-kernel GP regression fixture."""

    bound: BoundModel
    parameter: str
    x: jax.Array
    y: jax.Array
    f_true: jax.Array
    covariance: jax.Array
    chol: jax.Array
    obs_sd: float


@dataclass(frozen=True)
class NonCenteredKnownScaleFixture:
    """Bound non-centered known-scale hierarchy for analytic validation."""

    bound: BoundModel
    y_matrix: jax.Array
    prior_loc: float
    prior_scale: float
    group_scale: float
    obs_scale: float


@dataclass(frozen=True)
class StudentTLocationFixture:
    """Bound Student-t location model and grid-reference metadata."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    nu: float
    prior_loc: float
    prior_scale: float
    obs_scale: float


def eight_schools_fixture() -> EightSchoolsFixture:
    """Return the smoke fixture for a non-centered eight-schools model."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import HalfNormal, Normal

    @model
    class EightSchoolsNonCentered:
        """Non-centered eight-schools hierarchy with a positive population scale."""

        n_schools = Data()
        sigma = Data()

        mu = Param(Normal(0.0, 5.0))
        tau = Param(HalfNormal(5.0), constraint=Positive())
        z = Param(Normal(0.0, 1.0), size=n_schools)
        theta = mu + tau * z
        y = Observed(Normal(theta, sigma))

    y = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    return EightSchoolsFixture(
        bound=bind_model(EightSchoolsNonCentered, n_schools=8, sigma=sigma, y=y),
        y=y,
        sigma=sigma,
    )


def exponential_rate_fixture(
    *,
    seed: int = 3,
    sample_count: int = 400,
    true_rate: float = 2.0,
    prior_scale: float = 2.0,
) -> ExponentialRateFixture:
    """Return a constant-rate Exponential fixture."""
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Exponential, HalfNormal

    @model
    class ExponentialRate:
        """Constant-rate Exponential observations with a positive rate prior."""

        rate = Param(HalfNormal(prior_scale), constraint=Positive())
        y = Observed(Exponential(rate))

    y = jax.random.exponential(jax.random.PRNGKey(seed), (sample_count,)) / true_rate
    return ExponentialRateFixture(
        bound=bind_model(ExponentialRate, y=y),
        parameter="rate",
        y=y,
        prior_scale=prior_scale,
        true_rate=true_rate,
    )


def robust_regression_fixture() -> RobustRegressionFixture:
    """Return a robust linear-regression smoke fixture."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import HalfNormal, Normal, StudentT

    @model
    class RobustLinearRegression:
        """Linear regression with heavy-tailed Student-t observation noise."""

        nu = Data()
        x = Data()

        alpha = Param(Normal(0.0, 5.0))
        beta = Param(Normal(0.0, 5.0))
        sigma = Param(HalfNormal(5.0), constraint=Positive())
        mu = alpha + beta * x
        y = Observed(StudentT(nu, mu, sigma))

    key = jax.random.PRNGKey(7)
    noise_key, outlier_key = jax.random.split(key)
    x = jnp.linspace(-3.0, 3.0, 80)
    y_clean = 1.0 + 2.0 * x + 0.5 * jax.random.normal(noise_key, (80,))
    outliers = 15.0 * jax.random.bernoulli(outlier_key, 0.1, (80,)).astype(jnp.float32)
    y = y_clean + outliers
    nu = 4.0
    return RobustRegressionFixture(
        bound=bind_model(RobustLinearRegression, nu=nu, x=x, y=y),
        x=x,
        y=y,
        nu=nu,
    )


def multivariate_normal_likelihood_fixture() -> MultivariateNormalLikelihoodFixture:
    """Return a single-observation multivariate-normal likelihood fixture."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class MultivariateNormalLikelihood:
        """Single multivariate observation with a known Cholesky factor."""

        n_dim = Data()
        chol = Data()

        mu = Param(Normal(0.0, 10.0), size=n_dim)
        y = Observed(MultivariateNormal(mu, chol))

    covariance = jnp.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    chol = jnp.linalg.cholesky(covariance)
    y = jnp.array([1.5, -0.5, 2.0])
    return MultivariateNormalLikelihoodFixture(
        bound=bind_model(MultivariateNormalLikelihood, n_dim=3, chol=chol, y=y),
        parameter="mu",
        y=y,
        prior_mean=jnp.zeros((3,)),
        prior_scale=10.0,
        covariance=covariance,
        chol=chol,
    )


def rbf_covariance(x: jax.Array, *, lengthscale: float, amplitude: float) -> jax.Array:
    """Return an RBF covariance matrix for one-dimensional inputs."""
    diff = x[:, None] - x[None, :]
    return amplitude**2 * jnp.exp(-0.5 * (diff / lengthscale) ** 2)


def fixed_kernel_gp_fixture() -> FixedKernelGpFixture:
    """Return a fixed-kernel Gaussian-process regression fixture."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class GaussianProcessRegression:
        """Fixed-kernel GP latent with Normal observation noise."""

        n = Data()
        chol = Data()
        obs_sd = Data()

        f = Param(MultivariateNormal(0.0, chol), size=n)
        y = Observed(Normal(f, obs_sd))

    n = 30
    x = jnp.linspace(-3.0, 3.0, n)
    covariance = rbf_covariance(x, lengthscale=1.0, amplitude=1.0) + 1e-6 * jnp.eye(n)
    chol = jnp.linalg.cholesky(covariance)
    key = jax.random.PRNGKey(17)
    latent_key, noise_key = jax.random.split(key)
    f_true = chol @ jax.random.normal(latent_key, (n,))
    obs_sd = 0.2
    y = f_true + obs_sd * jax.random.normal(noise_key, (n,))
    return FixedKernelGpFixture(
        bound=bind_model(GaussianProcessRegression, n=n, chol=chol, obs_sd=obs_sd, y=y),
        parameter="f",
        x=x,
        y=y,
        f_true=f_true,
        covariance=covariance,
        chol=chol,
        obs_sd=obs_sd,
    )


def non_centered_known_scale_fixture() -> NonCenteredKnownScaleFixture:
    """Return a known-scale non-centered hierarchy for analytic validation."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import Normal

    prior_loc = 0.0
    prior_scale = 2.0
    group_scale = 1.0
    obs_scale = 0.5

    @model
    class NonCenteredHierarchicalNormal:
        """Known-scale non-centered Normal-Normal hierarchy."""

        n_groups = Data()
        group_idx = Data()

        mu_pop = Param(Normal(prior_loc, prior_scale))
        z = Param(Normal(0.0, 1.0), size=n_groups)
        theta = mu_pop + group_scale * z
        y = Observed(Normal(theta[group_idx], obs_scale))

    y_matrix = jnp.array([[1.0, 1.5], [2.0, 2.5]])
    group_idx = jnp.repeat(jnp.arange(2), 2)
    return NonCenteredKnownScaleFixture(
        bound=bind_model(
            NonCenteredHierarchicalNormal,
            n_groups=2,
            group_idx=group_idx,
            y=y_matrix.reshape(-1),
        ),
        y_matrix=y_matrix,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
        group_scale=group_scale,
        obs_scale=obs_scale,
    )


def student_t_location_fixture() -> StudentTLocationFixture:
    """Return a Student-t known-scale location model for grid validation."""
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal, StudentT

    nu = 4.0
    prior_loc = 0.0
    prior_scale = 2.0
    obs_scale = 1.0

    @model
    class StudentTLocation:
        """Scalar Student-t location model with known degrees of freedom and scale."""

        mu = Param(Normal(prior_loc, prior_scale))
        y = Observed(StudentT(nu, mu, obs_scale))

    y = jnp.array([-0.5, 0.25, 1.0, 1.25, 2.0])
    return StudentTLocationFixture(
        bound=bind_model(StudentTLocation, y=y),
        parameter="mu",
        y=y,
        nu=nu,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
        obs_scale=obs_scale,
    )
