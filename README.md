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

    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


x_data = jnp.linspace(-3.0, 3.0, 50)
y_data = 2.0 + 0.5 * x_data

bound = LinearRegression.bind(x=x_data, y=y_data)
result = sample(bound, seed=42, num_warmup=200, num_samples=500)

rhat_values = rhat(result.samples)
ess_values = ess(result.samples)
```

`result.samples` maps parameter names to JAX arrays with shape
`(1, num_samples, *param_shape)`. The leading dimension is the chain dimension;
currently sampling is single-chain NUTS.

## Model declarations

The declaration language has three core node types:

- `Param(distribution, constraint=None, size=None)` declares a latent stochastic
  value. It contributes a prior term to the log density and is sampled by NUTS.
- `Data()` declares known input with no likelihood contribution.
- `Observed(distribution)` declares known input with a likelihood contribution.
  A model may contain one or more observed likelihood sites.

Deterministic expressions are written directly in the class body:

```python
mu = alpha + beta * x
```

These expressions are symbolic during declaration and are evaluated by the
compiler after concrete data and parameter values are available.

## Hierarchical parameters

A parameter can have a static or data-dependent size:

```python
@model
class HierarchicalRegression:
    n_groups = Data()
    group_idx = Data()
    x = Data()

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
    n = Data()
    x_sd = Data()
    y_sd = Data()

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
