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

Data fixtures live in `data/` and are shared by the Stan log-density and
posterior-summary scripts.

Run the optional checks from the repository root:

```bash
uv run --script scripts/check_stan_log_density_reference.py
uv run --script scripts/check_stan_posterior_reference.py
uv run --script scripts/stress_stan_posterior_reference.py --runs 50
```

The log-density check compares jaxstan's unconstrained compiled log density to
CmdStan's log probability at equivalent fixed parameter values. CmdStanPy's
`log_prob` reports densities up to parameter-independent constants, so the
script restores the known Normal constants for these fixed fixtures before
comparison. For constrained parameters, CmdStan receives constrained values while
jaxstan receives the corresponding unconstrained values; both include the
Jacobian adjustment.

The posterior check runs Stan and jaxstan on the same fixed data and compares
posterior means using the combined Monte Carlo standard error. The stress script
repeats this comparison over configurable seeds and reports sampling time
summaries for both systems.
