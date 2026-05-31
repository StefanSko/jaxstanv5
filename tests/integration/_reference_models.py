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
    """Bound non-centered eight-schools model."""

    bound: BoundModel


@dataclass(frozen=True)
class ExponentialRateFixture:
    """Bound Exponential-rate model and data needed for references."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    prior_scale: float


@dataclass(frozen=True)
class PoissonLogRateFixture:
    """Bound scalar log-rate Poisson model and data needed for references."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    prior_loc: float
    prior_scale: float


@dataclass(frozen=True)
class BinomialLogisticFixture:
    """Bound scalar logistic Binomial model and data needed for references."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    trials: jax.Array
    prior_loc: float
    prior_scale: float


@dataclass(frozen=True)
class BetaBinomialLogisticFixture:
    """Bound scalar logistic Beta-binomial model and data needed for references."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    trials: jax.Array
    concentration: float
    prior_loc: float
    prior_scale: float


@dataclass(frozen=True)
class RobustRegressionFixture:
    """Bound robust linear regression with Student-t likelihood."""

    bound: BoundModel


@dataclass(frozen=True)
class HierarchicalPoissonFixture:
    """Bound non-centered hierarchical Poisson varying-slopes model."""

    bound: BoundModel
    y: jax.Array
    x: jax.Array
    group_idx: jax.Array
    n_groups: int


@dataclass(frozen=True)
class HierarchicalBinomialFixture:
    """Bound non-centered hierarchical Binomial varying-slopes model."""

    bound: BoundModel
    y: jax.Array
    trials: jax.Array
    x: jax.Array
    group_idx: jax.Array
    n_groups: int


@dataclass(frozen=True)
class HierarchicalBetaBinomialFixture:
    """Bound non-centered hierarchical Beta-binomial varying-slopes model."""

    bound: BoundModel
    y: jax.Array
    trials: jax.Array
    x: jax.Array
    group_idx: jax.Array
    n_groups: int


@dataclass(frozen=True)
class MultivariateNormalLikelihoodFixture:
    """Bound single-observation multivariate-normal likelihood model."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    prior_mean: jax.Array
    prior_scale: float
    covariance: jax.Array


@dataclass(frozen=True)
class FixedKernelGpFixture:
    """Bound fixed-kernel GP regression fixture."""

    bound: BoundModel
    parameter: str
    y: jax.Array
    covariance: jax.Array
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
    )


def poisson_log_rate_fixture() -> PoissonLogRateFixture:
    """Return a scalar log-rate Poisson fixture for grid validation."""
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.distributions import Normal, Poisson
    from jaxstanv5.math import exp

    prior_loc = 0.0
    prior_scale = 1.0

    @model
    class PoissonLogRate:
        """Scalar log-rate Poisson observations."""

        eta = Param(Normal(prior_loc, prior_scale))
        y = Observed(Poisson(exp(eta)))

    y = jnp.array([0.0, 1.0, 3.0, 2.0, 4.0, 1.0, 2.0, 3.0])
    return PoissonLogRateFixture(
        bound=bind_model(PoissonLogRate, y=y),
        parameter="eta",
        y=y,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def binomial_logistic_fixture() -> BinomialLogisticFixture:
    """Return a scalar logistic Binomial fixture for grid validation."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import Binomial, Normal
    from jaxstanv5.math import sigmoid

    prior_loc = 0.0
    prior_scale = 1.0

    @model
    class BinomialLogistic:
        """Scalar logistic Binomial observations."""

        eta = Param(Normal(prior_loc, prior_scale))
        trials = Data()
        y = Observed(Binomial(trials, sigmoid(eta)))

    trials = jnp.array([5.0, 8.0, 10.0, 12.0, 7.0, 9.0])
    y = jnp.array([1.0, 3.0, 5.0, 8.0, 2.0, 6.0])
    return BinomialLogisticFixture(
        bound=bind_model(BinomialLogistic, trials=trials, y=y),
        parameter="eta",
        y=y,
        trials=trials,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
    )


def beta_binomial_logistic_fixture() -> BetaBinomialLogisticFixture:
    """Return a scalar logistic Beta-binomial fixture for grid validation."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import BetaBinomial, Normal
    from jaxstanv5.math import sigmoid

    prior_loc = 0.0
    prior_scale = 1.0
    concentration = 12.0

    @model
    class BetaBinomialLogistic:
        """Scalar logistic Beta-binomial observations with fixed concentration."""

        eta = Param(Normal(prior_loc, prior_scale))
        trials = Data()
        concentration = Data()
        p = sigmoid(eta)
        a = p * concentration
        b = (1.0 - p) * concentration
        y = Observed(BetaBinomial(trials, a, b))

    trials = jnp.array([5.0, 8.0, 10.0, 12.0, 7.0, 9.0])
    y = jnp.array([1.0, 4.0, 5.0, 9.0, 1.0, 7.0])
    return BetaBinomialLogisticFixture(
        bound=bind_model(BetaBinomialLogistic, trials=trials, concentration=concentration, y=y),
        parameter="eta",
        y=y,
        trials=trials,
        concentration=concentration,
        prior_loc=prior_loc,
        prior_scale=prior_scale,
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
    )


def hierarchical_poisson_varying_slopes_fixture() -> HierarchicalPoissonFixture:
    """Return a hierarchical Poisson varying-slopes smoke fixture."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import HalfNormal, Normal, Poisson
    from jaxstanv5.math import exp

    @model
    class HierarchicalPoissonVaryingSlopes:
        """Non-centered Poisson log-rate model with varying intercepts and slopes."""

        n_groups = Data()
        group_idx = Data()
        x = Data()

        alpha_pop = Param(Normal(0.0, 0.5))
        beta_pop = Param(Normal(0.0, 0.5))
        sigma_alpha = Param(HalfNormal(0.4), constraint=Positive())
        sigma_beta = Param(HalfNormal(0.4), constraint=Positive())

        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        eta = alpha[group_idx] + beta[group_idx] * x
        y = Observed(Poisson(exp(eta)))

    n_groups = 4
    observations_per_group = 10
    group_idx = jnp.repeat(jnp.arange(n_groups), observations_per_group)
    x = jnp.tile(jnp.linspace(-1.0, 1.0, observations_per_group), n_groups)
    alpha_true = jnp.array([-0.25, 0.05, 0.35, -0.1])
    beta_true = jnp.array([0.25, -0.2, 0.3, 0.05])
    rate = jnp.exp(alpha_true[group_idx] + beta_true[group_idx] * x)
    y = jax.random.poisson(jax.random.PRNGKey(23), rate)
    return HierarchicalPoissonFixture(
        bound=bind_model(
            HierarchicalPoissonVaryingSlopes,
            n_groups=n_groups,
            group_idx=group_idx,
            x=x,
            y=y,
        ),
        y=y,
        x=x,
        group_idx=group_idx,
        n_groups=n_groups,
    )


def hierarchical_binomial_logistic_varying_slopes_fixture() -> HierarchicalBinomialFixture:
    """Return a hierarchical Binomial logistic varying-slopes smoke fixture."""
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Binomial, HalfNormal, Normal
    from jaxstanv5.math import sigmoid

    @model
    class HierarchicalBinomialLogisticVaryingSlopes:
        """Non-centered Binomial logit-probability model with varying slopes."""

        n_groups = Data()
        group_idx = Data()
        x = Data()
        trials = Data()

        alpha_pop = Param(Normal(0.0, 1.0))
        beta_pop = Param(Normal(0.0, 1.0))
        sigma_alpha = Param(HalfNormal(0.5), constraint=Positive())
        sigma_beta = Param(HalfNormal(0.5), constraint=Positive())

        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        eta = alpha[group_idx] + beta[group_idx] * x
        y = Observed(Binomial(trials, sigmoid(eta)))

    n_groups = 4
    observations_per_group = 12
    group_idx = jnp.repeat(jnp.arange(n_groups), observations_per_group)
    x = jnp.tile(jnp.linspace(-1.25, 1.25, observations_per_group), n_groups)
    trials = 10 + (jnp.arange(n_groups * observations_per_group) % 4)
    alpha_true = jnp.array([-0.8, -0.25, 0.4, 0.85])
    beta_true = jnp.array([0.75, -0.35, 0.2, -0.6])
    probs = jax.nn.sigmoid(alpha_true[group_idx] + beta_true[group_idx] * x)
    y = jax.random.binomial(jax.random.PRNGKey(31), n=trials, p=probs)
    return HierarchicalBinomialFixture(
        bound=bind_model(
            HierarchicalBinomialLogisticVaryingSlopes,
            n_groups=n_groups,
            group_idx=group_idx,
            x=x,
            trials=trials,
            y=y,
        ),
        y=y,
        trials=trials,
        x=x,
        group_idx=group_idx,
        n_groups=n_groups,
    )


def hierarchical_beta_binomial_logistic_varying_slopes_fixture() -> HierarchicalBetaBinomialFixture:
    """Return a hierarchical Beta-binomial logistic varying-slopes smoke fixture."""
    from math import log

    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import BetaBinomial, HalfNormal, Normal
    from jaxstanv5.math import exp, sigmoid

    @model
    class HierarchicalBetaBinomialLogisticVaryingSlopes:
        """Non-centered overdispersed Binomial model with varying slopes."""

        n_groups = Data()
        group_idx = Data()
        x = Data()
        trials = Data()

        alpha_pop = Param(Normal(0.0, 1.0))
        beta_pop = Param(Normal(0.0, 1.0))
        sigma_alpha = Param(HalfNormal(0.5), constraint=Positive())
        sigma_beta = Param(HalfNormal(0.5), constraint=Positive())
        log_concentration = Param(Normal(log(20.0), 0.5))

        z_alpha = Param(Normal(0.0, 1.0), size=n_groups)
        z_beta = Param(Normal(0.0, 1.0), size=n_groups)

        alpha = alpha_pop + sigma_alpha * z_alpha
        beta = beta_pop + sigma_beta * z_beta
        eta = alpha[group_idx] + beta[group_idx] * x
        p = sigmoid(eta)
        concentration = exp(log_concentration)
        a = p * concentration
        b = (1.0 - p) * concentration
        y = Observed(BetaBinomial(trials, a, b))

    n_groups = 4
    observations_per_group = 12
    group_idx = jnp.repeat(jnp.arange(n_groups), observations_per_group)
    x = jnp.tile(jnp.linspace(-1.25, 1.25, observations_per_group), n_groups)
    trials = 10 + (jnp.arange(n_groups * observations_per_group) % 4)
    alpha_true = jnp.array([-0.8, -0.25, 0.4, 0.85])
    beta_true = jnp.array([0.75, -0.35, 0.2, -0.6])
    concentration_true = 18.0
    p = jax.nn.sigmoid(alpha_true[group_idx] + beta_true[group_idx] * x)
    beta_key, count_key = jax.random.split(jax.random.PRNGKey(41))
    latent_p = jax.random.beta(beta_key, p * concentration_true, (1.0 - p) * concentration_true)
    y = jax.random.binomial(count_key, n=trials, p=latent_p)
    return HierarchicalBetaBinomialFixture(
        bound=bind_model(
            HierarchicalBetaBinomialLogisticVaryingSlopes,
            n_groups=n_groups,
            group_idx=group_idx,
            x=x,
            trials=trials,
            y=y,
        ),
        y=y,
        trials=trials,
        x=x,
        group_idx=group_idx,
        n_groups=n_groups,
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
    )


def rbf_covariance(x: jax.Array, *, lengthscale: float, amplitude: float) -> jax.Array:
    """Return an RBF covariance matrix for one-dimensional inputs."""
    diff = x[:, None] - x[None, :]
    return amplitude**2 * jnp.exp(-0.5 * (diff / lengthscale) ** 2)


def fixed_kernel_gp_fixture(
    *,
    n: int = 30,
    lengthscale: float = 1.0,
    amplitude: float = 1.0,
    obs_sd: float = 0.2,
) -> FixedKernelGpFixture:
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

    x = jnp.linspace(-3.0, 3.0, n)
    covariance = rbf_covariance(x, lengthscale=lengthscale, amplitude=amplitude) + 1e-6 * jnp.eye(n)
    chol = jnp.linalg.cholesky(covariance)
    key = jax.random.PRNGKey(17)
    latent_key, noise_key = jax.random.split(key)
    f_true = chol @ jax.random.normal(latent_key, (n,))
    y = f_true + obs_sd * jax.random.normal(noise_key, (n,))
    return FixedKernelGpFixture(
        bound=bind_model(GaussianProcessRegression, n=n, chol=chol, obs_sd=obs_sd, y=y),
        parameter="f",
        y=y,
        covariance=covariance,
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
