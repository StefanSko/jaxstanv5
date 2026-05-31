# Test layout

Tests are grouped by purpose:

```text
tests/unit/         small tests for isolated primitives and internal transitions
tests/integration/  public workflows and cross-module behavior
```

Use `unit` for local correctness. Use `integration` when behavior depends on the
public model declaration, binding, compiler, inference, or diagnostics workflow.

`tests/integration/_validation.py` is private scaffolding for posterior
validation. It records the Normal-path harness stages: analytic references,
typed summaries, public multi-chain draws, standardized discrepancies,
constrained references, Stan comparisons, and separate SBC checks. Public
sampler results also expose NUTS diagnostics, including divergences, for warmup
and post-warmup sampling. Stan fixtures live in `reference/stan/`, and optional
SBC references live in `scripts/check_sbc_reference.py`,
`scripts/check_gp_sbc_reference.py`, `scripts/check_poisson_sbc_reference.py`,
`scripts/check_binomial_sbc_reference.py`,
`scripts/check_beta_binomial_sbc_reference.py`, and
`scripts/check_negative_binomial_sbc_reference.py`; they are exercised by
standalone scripts rather than the default pytest suite. The fixed-kernel GP also
has a projected Stan posterior script, `scripts/check_gp_stan_posterior_reference.py`,
which compares calibrated scalar projections of the latent vector. The
hierarchical count models have dedicated Stan posterior scripts for Poisson,
Binomial, Beta-binomial, and Negative-binomial hyperparameters.
