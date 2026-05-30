"""Aspirational workflow coverage across target Bayesian model families.

This module is a smoke test for the public workflow only:
``@model -> bind -> sample -> diagnostics``.  It intentionally avoids calibrated
posterior correctness claims; family-specific validation lives in dedicated
``test_*_validation.py`` modules and uses ``_validation.py`` references, SBC, or
Stan.

Model builders import target distributions lazily so missing capabilities fail
as individual tests instead of aborting collection.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random
from _helpers import bind_model

from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.inference import sample


def _eight_schools_model() -> object:
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

    return EightSchoolsNonCentered


def test_eight_schools_non_centered_workflow_smoke() -> None:
    y_data = jnp.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma_data = jnp.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

    bound = bind_model(
        _eight_schools_model(),
        n_schools=8,
        sigma=sigma_data,
        y=y_data,
    )
    result = sample(bound, seed=11, num_chains=4, num_warmup=1000, num_samples=1000)

    rhat_vals = rhat(result.samples)
    for name in ("mu", "tau", "z"):
        assert rhat_vals[name] < 1.1, f"R-hat too high for {name}"
    assert jnp.all(jnp.isfinite(result.samples["mu"]))
    assert jnp.all(result.samples["tau"] > 0.0)


def _exponential_rate_model() -> object:
    from jaxstanv5 import Observed, Param, model
    from jaxstanv5.constraints import Positive
    from jaxstanv5.distributions import Exponential, HalfNormal

    @model
    class ExponentialRate:
        """Constant-rate Exponential observations with a positive rate prior."""

        rate = Param(HalfNormal(2.0), constraint=Positive())
        y = Observed(Exponential(rate))

    return ExponentialRate


def test_exponential_rate_workflow_smoke() -> None:
    rate_true = 2.0
    key = jax.random.PRNGKey(3)
    y_data = jax.random.exponential(key, (400,)) / rate_true

    bound = bind_model(_exponential_rate_model(), y=y_data)
    result = sample(bound, seed=5, num_chains=4, num_warmup=500, num_samples=1000)

    assert rhat(result.samples)["rate"] < 1.1
    assert ess(result.samples)["rate"] > 200
    assert jnp.all(jnp.isfinite(result.samples["rate"]))
    assert jnp.all(result.samples["rate"] > 0.0)


def _robust_regression_model() -> object:
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

    return RobustLinearRegression


def test_robust_regression_workflow_smoke() -> None:
    alpha_true, beta_true = 1.0, 2.0
    key = jax.random.PRNGKey(7)
    noise_key, outlier_key = jax.random.split(key)

    x_data = jnp.linspace(-3.0, 3.0, 80)
    clean = alpha_true + beta_true * x_data + 0.5 * jax.random.normal(noise_key, (80,))
    outliers = 15.0 * jax.random.bernoulli(outlier_key, 0.1, (80,)).astype(jnp.float32)
    y_data = clean + outliers

    bound = bind_model(_robust_regression_model(), nu=4.0, x=x_data, y=y_data)
    result = sample(bound, seed=9, num_chains=4, num_warmup=800, num_samples=1000)

    rhat_vals = rhat(result.samples)
    for name in ("alpha", "beta", "sigma"):
        assert rhat_vals[name] < 1.1, f"R-hat too high for {name}"
    assert jnp.all(result.samples["sigma"] > 0.0)


def _multivariate_normal_likelihood_model() -> object:
    from jaxstanv5 import Data, Observed, Param, model
    from jaxstanv5.distributions import MultivariateNormal, Normal

    @model
    class MultivariateNormalLikelihood:
        """Single multivariate observation with a known Cholesky factor."""

        n_dim = Data()
        chol = Data()

        mu = Param(Normal(0.0, 10.0), size=n_dim)
        y = Observed(MultivariateNormal(mu, chol))

    return MultivariateNormalLikelihood


def test_multivariate_normal_likelihood_workflow_smoke() -> None:
    cov = jnp.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
    chol = jnp.linalg.cholesky(cov)
    y_obs = jnp.array([1.5, -0.5, 2.0])

    bound = bind_model(_multivariate_normal_likelihood_model(), n_dim=3, chol=chol, y=y_obs)
    result = sample(bound, seed=13, num_chains=4, num_warmup=500, num_samples=1000)

    assert rhat(result.samples)["mu"] < 1.1
    assert jnp.all(jnp.isfinite(result.samples["mu"]))


def _gaussian_process_regression_model() -> object:
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

    return GaussianProcessRegression


def _rbf_cholesky(x: jax.Array, lengthscale: float, amplitude: float) -> jax.Array:
    diff = x[:, None] - x[None, :]
    kernel = amplitude**2 * jnp.exp(-0.5 * (diff / lengthscale) ** 2)
    kernel = kernel + 1e-6 * jnp.eye(x.shape[0])
    return jnp.linalg.cholesky(kernel)


def test_gaussian_process_workflow_smoke() -> None:
    n = 30
    x_data = jnp.linspace(-3.0, 3.0, n)
    chol = _rbf_cholesky(x_data, lengthscale=1.0, amplitude=1.0)

    key = jax.random.PRNGKey(17)
    latent_key, noise_key = jax.random.split(key)
    f_true = chol @ jax.random.normal(latent_key, (n,))
    obs_sd = 0.2
    y_data = f_true + obs_sd * jax.random.normal(noise_key, (n,))

    bound = bind_model(
        _gaussian_process_regression_model(),
        n=n,
        chol=chol,
        obs_sd=obs_sd,
        y=y_data,
    )
    result = sample(bound, seed=19, num_chains=4, num_warmup=800, num_samples=1000)

    assert rhat(result.samples)["f"] < 1.15
    assert jnp.all(jnp.isfinite(result.samples["f"]))
