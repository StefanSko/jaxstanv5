# jaxstanv5

> **⚠️ This repository is archived, and the package was renamed.**
> jaxstanv5 is now **bayesjax**, living in the
> [bayescycle monorepo](https://github.com/StefanSko/bayescycle) at
> [`packages/bayesjax`](https://github.com/StefanSko/bayescycle/tree/main/packages/bayesjax)
> and published to PyPI as
> [`bayesjax`](https://pypi.org/project/bayesjax/). History here remains
> readable for old pins. Issues have been transferred to the monorepo.

`jaxstanv5` is the JAX/BlackJAX sampling backend for
[bayeswire](https://github.com/StefanSko/bayeswire) models.

It focuses on one workflow:

```text
declare a bayeswire model -> bind data -> sample with NUTS -> inspect basic diagnostics
```

It is intentionally small: no workflow platform, plotting layer, reporting
system, artifact store, or multi-algorithm inference API. The model
declaration language, the IR wire format, its normative spec, and the golden
conformance corpus live in bayeswire; jaxstanv5 consumes them and owns
binding, compiled log densities, constraint transforms and Jacobians, NUTS
via BlackJAX, essential diagnostics, simulation, and the
InferenceData-compatible schema.

## Quickstart

```bash
uv sync
```

Define a model with bayeswire, then bind and sample with jaxstanv5:

```python
import jax.numpy as jnp

from bayeswire import Data, Observed, Param, model
from bayeswire.constraints import Positive
from bayeswire.distributions import Normal, Truncated
from jaxstanv5 import bind_model
from jaxstanv5.diagnostics import ess, rhat
from jaxstanv5.inference import sample


@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

    x = Data.vector()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))


x_data = jnp.linspace(-3.0, 3.0, 50)
y_data = 2.0 + 0.5 * x_data

bound = bind_model(LinearRegression, {"x": x_data, "y": y_data})
result = sample(bound, seed=42, num_warmup=200, num_samples=500, num_chains=4)

rhat_values = rhat(result.samples)
ess_values = ess(result.samples)
sampling_divergences = result.diagnostics.sampling.is_divergent
```

`result.samples` maps parameter names to JAX arrays with shape
`(num_chains, num_samples, *param_shape)`. The leading dimension is the chain
dimension. Zero-sized declared parameters are preserved with their zero-length
parameter axis. If `num_chains` is omitted, sampling defaults to one chain.

`result.diagnostics` records NUTS diagnostics for warmup and post-warmup
sampling. Diagnostic arrays have shape `(num_chains, num_steps)`. `num_warmup`
and `num_samples` must both be at least 1. Use `target_acceptance_rate` on
`sample(...)` to tune the NUTS adaptation target.

## InferenceData-compatible schema

jaxstanv5 does not depend on ArviZ, xarray, netCDF, or zarr, and `sample(...)`
does not return an ArviZ object. For downstream exporters, use the typed schema
adapter:

```python
from jaxstanv5.interop.inferencedata import inferencedata_groups

schema = inferencedata_groups(bound, result)
```

It maps posterior draws, post-warmup NUTS diagnostics, observed data, constant
data, and declared `Dim(...)` metadata into InferenceData-compatible groups and
dimension names. See [`docs/inferencedata-compatibility.md`](docs/inferencedata-compatibility.md).

## The language lives in bayeswire

Model declarations (`@model`, `Param`, `Data`, `Observed`,
`PartiallyObserved`, `Dim`), the distributions and constraints metadata, the
symbolic math namespace, IR serialization (`bayeswire.ir`), and the dimension
sidecar all belong to [bayeswire](https://github.com/StefanSko/bayeswire) —
see its README and `spec/` for the declaration language and the
`bayeswire_ir` v1 wire format. jaxstanv5 pins bayeswire by exact version;
the pin bump diff is the compatibility review.

jaxstanv5's authoring-facing surface is the explicit backend transition:

```python
from bayeswire.ir import bindable_from_meta, meta_from_dict
from bayeswire.model import dimension_metadata_from_dict
from jaxstanv5 import bind_model

meta = meta_from_dict(document)
dimensions = dimension_metadata_from_dict(dimension_document)
rebuilt = bindable_from_meta(meta, dimensions=dimensions)
bound = bind_model(rebuilt, values)
```

Dynamic workflow adapters should use `bayeswire.model.model_meta(...)`,
`bayeswire.model.is_model_class(...)`, and `jaxstanv5.bind_model(...)`.
The consume-conformance test in `tests/integration/` proves this backend
evaluates the bayeswire corpus fixtures within the spec's tolerance policy.

## Backend semantics notes

The declaration language is documented in bayeswire. Backend-relevant
semantics when sampling with jaxstanv5:

- Discrete distributions such as `Poisson`, `Binomial`, `BetaBinomial`, and
  `NegativeBinomial` are valid observed likelihoods, but not latent
  `Param(...)` priors because NUTS samples continuous parameters only.
- Constraints define NUTS transforms and Jacobians. When a prior needs
  truncation normalization, make it explicit with
  `Truncated(base, lower=..., upper=...)` and a matching constraint.
- `OrderedLogistic(eta, cutpoints)` uses zero-based observed labels: with `K`
  cutpoints, valid categories are `0..K`.
- For partially observed continuous vectors,
  `jaxstanv5.data.PartialVector.from_nan(...)` converts a NaN-masked vector
  into the explicit index partition the model consumes. Missing coordinates
  are returned in `result.samples["y"]` in `missing_idx` order. Discrete
  missing latents are not supported by NUTS.

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
    sigma_alpha = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
    sigma_beta = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

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
    sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

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

For backend-boundary refactors, also run a small statistical smoke set covering
both Stan and SBC references, for example:

```bash
uv run --script scripts/check_poisson_stan_posterior_reference.py
uv run --script scripts/check_beta_regression_stan_posterior_reference.py
uv run --script scripts/check_binomial_sbc_reference.py
uv run --script scripts/check_ordinal_sbc_reference.py
```

Important internal invariants are documented in
[`docs/invariants.md`](docs/invariants.md).
