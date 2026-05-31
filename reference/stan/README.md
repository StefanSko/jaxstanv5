# Stan reference fixtures

This directory contains fixed Stan models and data used by optional reference
validation scripts. They are not part of the default pytest suite and are not a
runtime dependency of `jaxstanv5`.

Models:

- `models/normal_known_scale.stan`
  - `mu ~ normal(prior_loc, prior_scale)`
  - `y ~ normal(mu, obs_scale)`
- `models/positive_scale_normal.stan`
  - `sigma ~ normal(prior_loc, prior_scale)`, with `sigma > 0`
  - `y ~ normal(0, sigma)`
- `models/exponential_rate.stan`
  - `rate ~ normal(0, prior_scale)`, with `rate > 0`
  - `y ~ exponential(rate)`
- `models/student_t_location.stan`
  - `mu ~ normal(prior_loc, prior_scale)`
  - `y ~ student_t(nu, mu, obs_scale)`
- `models/hierarchical_poisson_varying_slopes.stan`
  - non-centered varying intercepts and slopes
  - `y ~ poisson_log(alpha[group] + beta[group] * x)`
- `models/hierarchical_binomial_logistic_varying_slopes.stan`
  - non-centered varying intercepts and slopes
  - `y ~ binomial_logit(trials, alpha[group] + beta[group] * x)`
- `models/hierarchical_beta_binomial_logistic_varying_slopes.stan`
  - non-centered varying intercepts and slopes with log concentration
  - `y ~ beta_binomial(trials, p * concentration, (1 - p) * concentration)`
- `models/hierarchical_beta_regression_logistic_varying_slopes.stan`
  - non-centered varying intercepts and slopes with log precision
  - `y ~ beta(mu * phi, (1 - mu) * phi)`
- `models/hierarchical_negative_binomial_log_rate_varying_slopes.stan`
  - non-centered varying intercepts and slopes with log overdispersion
  - `y ~ neg_binomial_2(exp(alpha[group] + beta[group] * x), overdispersion)`
- `models/multivariate_normal_likelihood.stan`
  - `mu ~ normal(0, prior_scale)`
  - `y ~ multi_normal_cholesky(mu, chol)`
- `models/fixed_kernel_gp.stan`
  - `f ~ multi_normal_cholesky(0, chol)`
  - `y ~ normal(f, obs_sd)`

Data fixtures live in `data/` and are shared by the Stan log-density and
posterior-summary scripts where applicable. `data/fixed_kernel_gp_n8.json` is a
larger fixed-kernel GP posterior fixture used by projected Stan comparisons.

Run the optional checks from the repository root:

```bash
uv run --script scripts/check_stan_log_density_reference.py
uv run --script scripts/check_stan_posterior_reference.py
uv run --script scripts/check_poisson_stan_posterior_reference.py
uv run --script scripts/check_binomial_stan_posterior_reference.py
uv run --script scripts/check_beta_binomial_stan_posterior_reference.py
uv run --script scripts/check_beta_regression_stan_posterior_reference.py
uv run --script scripts/check_negative_binomial_stan_posterior_reference.py
uv run --script scripts/check_gp_stan_posterior_reference.py
uv run --script scripts/stress_stan_posterior_reference.py --runs 50
```

The log-density check compares jaxstan's unconstrained compiled log-density
**differences** to CmdStan's log-probability differences at equivalent fixed
parameter values. CmdStanPy's `log_prob` reports densities up to
parameter-independent constants, and those constants cancel in differences. For
constrained parameters, CmdStan receives constrained values while jaxstan
receives the corresponding unconstrained values; both include the Jacobian
adjustment.

The posterior checks run Stan and jaxstan on the same fixed data and compare
posterior means using the combined Monte Carlo standard error. Scalar cases are
compared directly. The fixed-kernel GP posterior script compares fixed linear
projections of the latent vector (`f[0]`, `f[n // 2]`, `mean(f)`, and
`f[-1] - f[0]`) so vector posterior behavior is checked through calibrated
scalar summaries. The hierarchical count/proportion posterior scripts compare
scalar hyperparameters from one shared Stan/jaxstan run. The posterior scripts align
jaxstan's `target_acceptance_rate` with Stan's `adapt_delta`; scalar posterior
checks default to `0.95`, while hierarchical count and GP scripts default to
`0.90` for their geometries. The stress script repeats scalar comparisons over
configurable seeds and reports NUTS diagnostics (divergences, acceptance rates,
and integration-step counts) and sampling time summaries for both systems.
