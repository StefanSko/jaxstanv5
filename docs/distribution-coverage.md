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
- **Poisson count likelihood** with symbolic log-rate construction.
- **Hierarchical Poisson varying slopes** (non-centered group effects with a
  symbolic exponential link).
- **Multivariate** (vector-valued likelihood).
- **Gaussian process** (multivariate-normal latent with a fixed-kernel
  Cholesky factor supplied as data).
- **Measurement error** (already shipped; kept green).

## Distribution invariants

- Every distribution implements `log_prob`. Element-wise distributions return
  one log-density/log-mass per element; `MultivariateNormal` is event-wise and
  returns one log-density per event vector.
- The compiler aggregates every site as `jnp.sum(dist.log_prob(value))`. This
  holds for element-wise distributions and for a single event-wise vector.
- A distribution used as a prior for an unconstrained parameter must implement
  `sample`, `batch_shape`, and `event_shape` (it is `SampleableDistribution`).
  Prior simulation distinguishes iid sample dimensions, distribution batch
  dimensions, and event dimensions; event-wise priors such as
  `MultivariateNormal` are supported.
- A distribution used as a prior for an interval-constrained parameter must also
  implement `cdf` and `icdf` (it is `InverseCdfDistribution`), because prior
  simulation uses inverse-CDF restricted sampling.
- A distribution used only as a likelihood, or only as an unconstrained prior,
  is not required to implement `cdf`/`icdf`.
- Discrete distributions are valid for `Observed(...)` likelihoods and
  prior-predictive observed simulation, but not for latent `Param(...)` priors
  because NUTS samples continuous parameters only.

## Sampling vs. simulation

- NUTS sampling depends only on the compiled log-density and a zero
  initialization; it does not require prior sampling. Multivariate, GP, and
  observed-count models therefore sample without changes to latent prior
  simulation.
- Prior simulation (`simulation/core.py`) treats a model parameter's resolved
  shape as the full constrained value shape. It derives the iid sample shape by
  removing the distribution's `batch_shape + event_shape` suffix before calling
  `sample`. This keeps vector scalar-event priors and single vector-event priors
  distinct even when their final value shapes are identical.

## Validation invariants

- Each model family is validated against a reference: analytic (conjugate),
  numerical (1-D grid), or simulation-based calibration where prior sampling is
  available.
- Always-on tests use fast analytic or numerical references. Stan and SBC remain
  out-of-band (slow), consistent with the existing validation plan.
- Standardized discrepancies (signed z / k_min) compare posterior summaries to
  references within Monte Carlo standard error.
- Vector-valued GP validation uses fixed linear projections (`f[0]`,
  `f[n // 2]`, `mean(f)`, and `f[-1] - f[0]`) for script-based Stan posterior
  comparison and projected SBC. Each projection is a scalar posterior functional
  and is compared with the same MCSE-calibrated machinery as scalar parameters.
- Hierarchical Poisson validation has an always-on scalar grid reference, a
  workflow smoke test for the varying-slopes model, Stan log-density/posterior
  scripts, and an optional SBC script over scalar hyperparameters.
