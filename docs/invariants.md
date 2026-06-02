# Invariants

Core invariants that should remain true as the codebase changes.

## Scope

- Public workflow: define model -> bind data -> sample with NUTS.
- NUTS is the only inference algorithm.
- BlackJAX is internal.

## Declaration language

- `Param`, `Data`, and `Observed` are declarations.
- `Param(...)` is latent and contributes a prior term.
- `Data()` is known input and contributes no log-density term.
- `Observed(...)` is known input and contributes a likelihood term.
- A model has one or more stochastic declarations: `Param` or `Observed`.
- `Observed` nodes are optional; prior-only models are valid.
- Declaration aliases are invalid: one declaration object maps to one class
  attribute name.

## Phase boundaries

- Class-body syntax capture and resolved model metadata are separate phases.
- Class-body arithmetic, indexing, and supported `jaxstanv5.math` helpers create
  private deferred syntax, never final expression IR.
- Declaration expressions support Python scalar literals as constants; fixed
  non-scalar inputs must be represented explicitly as `Data()`.
- Raw JAX/NumPy functions are not declaration-language operations; supported
  symbolic math functions cross the declaration boundary through explicit helper
  nodes.
- `_resolve_model_declaration(...)` is the only transition from declaration
  symbols to named references.
- `bind(...)` is the transition from resolved model class to `BoundModel`.

## IR and module boundaries

- `_deferred.py` is private class-body syntax capture.
- `core.py` does not construct final expression IR.
- `expr.py` is resolved/final IR only.
- Final expression trees contain no declaration symbols, raw declarations, or
  deferred syntax tokens.
- Final expression trees may contain explicit unary operation nodes only for
  supported declaration-language unary operations such as `neg`, `exp`, and `sigmoid`.
- `ModelMeta` contains resolved metadata only.
- `BoundModel` contains no inference logic.

## Log density

- The compiler evaluates symbolic distribution arguments before calling
  `Distribution.log_prob(...)`.
- Log density = constraint Jacobians + parameter priors + all observed
  likelihood terms.
- Discrete distributions are observed-likelihood distributions only; NUTS
  parameters are continuous and discrete latent `Param(...)` declarations are
  rejected.
- Sampling returns constrained parameter values.
- Sampling result arrays have shape `(num_chains, num_samples, *param_shape)`.
- NUTS diagnostics are recorded separately for warmup and post-warmup sampling.
- Diagnostic arrays have shape `(num_chains, num_steps)`, where `num_steps` is
  `num_warmup` for warmup diagnostics and `num_samples` for sampling diagnostics.

## Simulation

- Prior and prior-predictive simulation distinguish iid sample dimensions,
  distribution batch dimensions, and event dimensions.
- Distribution samples have shape `iid_sample_shape + batch_shape + event_shape`.
- Discrete observed distributions may be sampled for prior-predictive
  simulation, but discrete latent parameters remain unsupported.
- A model parameter's resolved `param_shape` is the full constrained value shape,
  not necessarily the iid sample shape passed to a distribution.
- Prior simulation derives iid sample shape by removing the distribution's
  `batch_shape + event_shape` suffix from the resolved model value shape.
- Event-shaped priors, such as `MultivariateNormal`, are valid when their event
  shape is part of the resolved model value shape.
- `Interval(lower, upper)` and `UnitInterval()` represent finite open intervals
  only. Their transforms are explicit logit/scaled-logit bijections and their
  inverse-transform Jacobians are part of compiled latent log densities.
- Interval-constrained prior simulation is limited to scalar-event inverse-CDF
  distributions unless explicit multivariate constrained simulation is added.
- Ordered-vector prior simulation is limited to iid scalar-event priors. It samples
  constrained values by sorting iid prior draws along the last axis; it does not
  use the ordered unconstraining transform or a Jacobian because simulation draws
  directly from the normalized constrained prior.
