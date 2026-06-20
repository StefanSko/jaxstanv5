# InferenceData compatibility

jaxstanv5 exposes an InferenceData-compatible schema adapter without depending on
ArviZ, xarray, netCDF, or zarr.

The boundary is intentionally narrow:

```text
BoundModel + SamplerResult -> typed InferenceData-compatible groups
```

The adapter does not construct an `arviz.InferenceData` object and does not write
fit artifacts. Downstream workflow or visualization packages own those
dependencies and storage decisions.

## API

```python
from jaxstanv5.interop.inferencedata import inferencedata_groups

schema = inferencedata_groups(bound, result)
```

The returned value is an `InferenceDataGroups` dataclass with:

- `posterior`
- `sample_stats`
- `observed_data`
- `constant_data`
- `coords`

Each group contains named `InferenceDataVariable` values with explicit `dims` and
JAX-array `values`.

## Group mapping

| Source | Group | Variable mapping |
|---|---|---|
| `SamplerResult.samples` | `posterior` | constrained parameter draws |
| `SamplerResult.diagnostics.sampling` | `sample_stats` | post-warmup NUTS diagnostics |
| bound `Observed(...)` inputs | `observed_data` | observed likelihood values |
| bound `Data(...)` inputs | `constant_data` | declared non-stochastic inputs |

Posterior arrays keep the public jaxstanv5 layout:

```text
(chain, draw, *param_shape)
```

Observed and constant data arrays use their value shape without chain/draw axes.

## Sample stats names

The adapter maps post-warmup NUTS diagnostics to conventional InferenceData names:

| jaxstanv5 diagnostic | InferenceData-compatible variable |
|---|---|
| `is_divergent` | `diverging` |
| `acceptance_rate` | `acceptance_rate` |
| `num_integration_steps` | `n_steps` |
| `num_trajectory_expansions` | `tree_depth` |
| `energy` | `energy` |

Warmup diagnostics are intentionally not included in the MVP schema adapter.

## Dimensions and coordinates

`chain` and `draw` coordinates are generated as zero-based integer coordinates.

Declared `Dim(...)` labels and coordinates are preserved through `bind(...)` and
used when they match the bound value rank. For example:

```python
predictor = Dim("predictor", coords=("x1", "x2"))

@model
class Regression:
    beta = Param(Normal(0.0, 1.0), size=2, dims=(predictor,))
```

produces:

```text
posterior.beta(chain, draw, predictor)
coords["predictor"] == ("x1", "x2")
```

When a variable has no declared dimension metadata, the adapter generates stable
fallback dimension names and integer coordinates:

```text
theta(chain, draw, theta_dim_0)
coords["theta_dim_0"] == (0, 1, ...)
```

If the same dimension name is used with inconsistent axis sizes or conflicting
coordinates, the adapter raises `ValueError` rather than producing an ambiguous
schema.

## Non-goals

jaxstanv5 does not:

- import ArviZ;
- construct xarray datasets;
- write netCDF or zarr stores;
- make `sample(...)` return an ArviZ-like object;
- own plotting or workflow artifact lifecycles.

Those responsibilities belong to downstream workflow/export/visualization
packages.
