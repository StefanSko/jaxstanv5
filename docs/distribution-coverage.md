# Distribution coverage goal and invariants

This document records the target end-state for growing `jaxstanv5` from a
Normal/Uniform core into a small but real Bayesian distribution library, and the
invariants that must stay true while we get there.

The work is driven outside-in: one aspirational integration test
(`tests/integration/test_distribution_coverage.py`) declares the target models
and stays red until each model family is implemented through staged red->green
TDD.

## Goal

Each target model family can be:

1. declared with `@model`, `Param`, `Data`, `Observed`,
2. compiled to a NUTS log-density,
3. sampled with the public `sample(...)` API, and
4. validated through the staged harness in `tests/integration/_validation.py`
   (reference -> draws -> summaries -> standardized discrepancy; SBC and Stan
   where applicable).

Target model families:

- **Non-centered hierarchical** (eight-schools style reparameterization).
- **Hierarchical with learned scales** (positive-scale hyperpriors).
- **Robust regression** (Student-t likelihood).
- **Exponential likelihood** (positive-support observations).
- **Multivariate** (vector-valued likelihood).
- **Gaussian process** (multivariate-normal latent with a fixed-kernel
  Cholesky factor supplied as data).
- **Measurement error** (already shipped; kept green).

## Distribution invariants

- Every distribution implements `log_prob`. Element-wise distributions return
  one log-density per element; `MultivariateNormal` is event-wise and returns
  one log-density per event vector.
- The compiler aggregates every site as `jnp.sum(dist.log_prob(value))`. This
  holds for element-wise distributions and for a single event-wise vector.
- A distribution used as a prior for an unconstrained parameter must implement
  `sample` (it is `SampleableDistribution`).
- A distribution used as a prior for an interval-constrained parameter must also
  implement `cdf` and `icdf` (it is `InverseCdfDistribution`), because prior
  simulation uses inverse-CDF restricted sampling.
- A distribution used only as a likelihood, or only as an unconstrained prior,
  is not required to implement `cdf`/`icdf`.

## Sampling vs. simulation

- NUTS sampling depends only on the compiled log-density and a zero
  initialization; it does not require prior sampling. Multivariate and GP models
  therefore sample without changes to prior simulation.
- Prior simulation (`simulation/core.py`) currently assumes
  `sample_shape == param_shape` (independent element-wise draws). Event-wise
  distributions such as `MultivariateNormal` do not satisfy this, so SBC for
  multivariate models is deferred until simulation distinguishes event
  dimensions from sample dimensions. Until then, multivariate models are
  validated by analytic references and out-of-band Stan checks.

## Validation invariants

- Each model family is validated against a reference: analytic (conjugate),
  numerical (1-D grid), or simulation-based calibration where prior sampling is
  available.
- Always-on tests use fast analytic or numerical references. Stan and SBC remain
  out-of-band (slow), consistent with the existing validation plan.
- Standardized discrepancies (signed z / k_min) compare posterior summaries to
  references within Monte Carlo standard error.
