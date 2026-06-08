# jaxstanv5

`jaxstanv5` is a minimal declarative Bayesian modeling library for JAX.

It focuses on one workflow:

```text
define a model -> bind data -> sample with NUTS -> inspect basic diagnostics
```

It is intentionally small: no workflow platform, plotting layer, reporting
system, artifact store, or multi-algorithm inference API.

## Quickstart

Install dependencies:

```bash
uv sync
```

Define a model:

```python
import jax.numpy as jnp

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.distributions import Normal
from jaxstanv5.inference import sample


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())

    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


x_data = jnp.linspace(-3.0, 3.0, 50)
y_data = 2.0 + 0.5 * x_data

bound = LinearRegression.bind(x=x_data, y=y_data)
result = sample(bound, seed=42, num_warmup=200, num_samples=500, num_chains=4)

rhat_values = rhat(result.samples)
ess_values = ess(result.samples)
sampling_divergences = result.diagnostics.sampling.is_divergent
```

`result.samples` maps parameter names to JAX arrays with shape
`(num_chains, num_samples, *param_shape)`. The leading dimension is the chain
dimension. If `num_chains` is omitted, sampling defaults to one chain.

`result.diagnostics` records NUTS diagnostics for warmup and post-warmup
sampling. Diagnostic arrays have shape `(num_chains, num_steps)`. `num_warmup`
and `num_samples` must both be at least 1. Use `target_acceptance_rate` on
`sample(...)` to tune the NUTS adaptation target.

## Model declarations

The declaration language has three core node types:

- `Param(distribution, constraint=None, size=None)` declares a latent stochastic
  value. It contributes a prior term to the log density and is sampled by NUTS.
- `Data.scalar()`, `Data.vector(...)`, `Data.matrix(...)`, and
  `Data.array(...)` declare known inputs with no likelihood contribution.
- `Observed(distribution)` declares known input with a likelihood contribution.
  A model may contain one or more observed likelihood sites.
- `PartiallyObserved.vector(...)` declares one continuous random vector whose
  coordinates are partly fixed data and partly free NUTS coordinates.

Deterministic expressions are written directly in the class body:

```python
mu = alpha + beta * x
```

These expressions are symbolic during declaration and are evaluated by the
compiler after concrete data and parameter values are available. Python scalar
literals are valid expression constants. Fixed vector, matrix, or tensor inputs
must be declared with shaped `Data` helpers and passed to `bind(...)`, rather
than captured as closed-over array constants. A small symbolic math namespace is
available for supported nonlinear declarations:

```python
from jaxstanv5.distributions import Beta, Binomial, Poisson
from jaxstanv5.math import exp, sigmoid

rate = exp(alpha + beta * x)
y_count = Observed(Poisson(rate))

probability = sigmoid(alpha + beta * x)
y_successes = Observed(Binomial(trials, probability))
y_proportion = Observed(Beta(probability * concentration, (1.0 - probability) * concentration))
```

Declaration indexing supports integer scalar or data indexes and full-axis
slices, for example `alpha[group_idx]`, `x[:, 0]`, and `x[group_idx, 0]`.
General NumPy indexing forms such as partial slices, ellipses, new axes, and
boolean masks are not part of the declaration language.

Custom distributions can implement the `Distribution` protocol. If their
parameters include `Param`, `Data`, or symbolic expressions, define the custom
distribution as a dataclass so model declaration can resolve those fields.
Opaque non-dataclass distributions are allowed only with concrete parameters.

Data declarations are schema-only; they never construct values. Use
`Data.scalar()` for rank-0 inputs, `Data.vector()` or `Data.vector(n)` for
vectors, `Data.matrix()` or `Data.matrix(n, m)` for matrices, and
`Data.array(rank=...)` or `Data.array(shape=(...))` for higher-rank arrays.
Rank-only declarations validate rank while allowing any dimension sizes.

For partially observed continuous vectors, pass an explicit index partition rather
than a NaN-masked vector:

```python
from jaxstanv5 import PartiallyObserved
from jaxstanv5.data import PartialVector
from jaxstanv5.distributions import MultivariateNormal

partial_y = PartialVector.from_nan(raw_y)

@model
class PartialMvn:
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
```

`PartialVector.from_nan(...)` is only a preprocessing helper; the model consumes
explicit `observed_idx`, `missing_idx`, and `observed_values`. Missing coordinates
are returned in `result.samples["y"]` in `missing_idx` order. Discrete missing
latents are not supported by NUTS.

Use `jaxstanv5.math` helpers in model declarations, not raw `jax.numpy`
functions. Discrete distributions such as `Poisson`, `Binomial`,
`BetaBinomial`, and `NegativeBinomial` are valid observed likelihoods, but not
latent `Param(...)` priors because NUTS samples continuous parameters only.
Continuous bounded latent parameters can use `Interval(lower, upper)` or
`UnitInterval()` constraints. Ordered vector parameters can use `Ordered()`, for
example ordinal-logistic cutpoints. `OrderedLogistic(eta, cutpoints)` uses
zero-based observed labels: with `K` cutpoints, valid categories are `0..K`.

## Hierarchical parameters

A parameter can have a static or data-dependent size:

```python
@model
class HierarchicalRegression:
    n_groups = Data.scalar()
    group_idx = Data.vector()
    x = Data.vector()

    alpha_pop = Param(Normal(0.0, 1.0))
    beta_pop = Param(Normal(0.0, 1.0))
    sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
    sigma_beta = Param(Normal(0.0, 1.0), constraint=Positive())
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())

    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    beta = Param(Normal(beta_pop, sigma_beta), size=n_groups)

    mu = alpha[group_idx] + beta[group_idx] * x
    y = Observed(Normal(mu, sigma))
```

At bind time, `n_groups` is resolved to the shape of `alpha` and `beta`.

## Multiple observed likelihood sites

Use multiple `Observed(...)` declarations for measurement-error models or other
models with more than one observed likelihood factor.

```python
@model
class MeasurementErrorRegression:
    n = Data.scalar()
    x_sd = Data.vector(n)
    y_sd = Data.vector(n)

    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())

    x_true = Param(Normal(0.0, 1.0), size=n)
    mu = alpha + beta * x_true
    y_true = Param(Normal(mu, sigma), size=n)

    x_obs = Observed(Normal(x_true, x_sd))
    y_obs = Observed(Normal(y_true, y_sd))
```

Here:

- `x_true` and `y_true` are latent parameters.
- `x_sd` and `y_sd` are known inputs.
- `x_obs` and `y_obs` are bound data values that each add a likelihood term.

The compiled log density is:

```text
parameter priors + constraint Jacobians + all observed likelihood terms
```

## Reference checks

Prior and prior-predictive simulation are available through
`jaxstanv5.simulation.simulate_prior_predictive(...)` for supported model
fragments. It draws parameters and observed values with a leading simulation
axis, using fixed shaped `Data` values and optional observed-site shapes.

Optional Stan reference fixtures live in [`reference/stan/`](reference/stan/).
They are used by standalone scripts, not by the default pytest suite and not by
runtime code.

Run fixed-data Stan comparisons with:

```bash
uv run --script scripts/check_stan_log_density_reference.py
uv run --script scripts/check_stan_posterior_reference.py
uv run --script scripts/check_poisson_stan_posterior_reference.py
uv run --script scripts/check_binomial_stan_posterior_reference.py
uv run --script scripts/check_beta_binomial_stan_posterior_reference.py
uv run --script scripts/check_beta_regression_stan_posterior_reference.py
uv run --script scripts/check_negative_binomial_stan_posterior_reference.py
uv run --script scripts/check_ordinal_stan_posterior_reference.py
uv run --script scripts/stress_stan_posterior_reference.py --runs 50
```

The log-density script compares jaxstan's unconstrained compiled density against
CmdStan at equivalent parameter values, with known Normal constants restored for
these fixtures. The posterior script compares jaxstan and Stan posterior means
using combined MCSE-scaled discrepancies. The posterior scripts align jaxstan's
`target_acceptance_rate` with Stan's `adapt_delta` and default to `0.95`. The
stress script repeats the Stan posterior comparison across seeds and reports
sampling time summaries for both jaxstan and Stan.

Optional SBC checks run prior-predictive simulations, fit generated datasets,
and check posterior ranks:

```bash
uv run --script scripts/check_sbc_reference.py --case all
uv run --script scripts/check_poisson_sbc_reference.py
uv run --script scripts/check_binomial_sbc_reference.py
uv run --script scripts/check_beta_binomial_sbc_reference.py
uv run --script scripts/check_beta_regression_sbc_reference.py
uv run --script scripts/check_negative_binomial_sbc_reference.py
uv run --script scripts/check_ordinal_sbc_reference.py
```

A restricted raw-model adapter is also available for supported Normal models:

```bash
uv run --script scripts/check_sbc_reference.py \
  --model-file path/to/model.py:MyModel \
  --parameter mu \
  --observed-shape y=8
```

## Development

Run the full validation loop with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Important internal invariants are documented in
[`docs/invariants.md`](docs/invariants.md).
