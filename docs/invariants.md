# Invariants

Core invariants that should remain true as the codebase changes.

## Scope

- Public workflow: declare a bayeswire model -> bind data -> sample with NUTS.
- NUTS is the only inference algorithm.
- BlackJAX is internal.

## Authoring, IR, and backend boundaries

- The authoring/IR boundary is the **bayeswire package**. Model declarations,
  distribution and constraint metadata, `ModelMeta`, IR serialization, and
  the dimension sidecar live there; jaxstanv5 depends on bayeswire pinned by
  exact version and contains no declaration-language semantics of its own.
- The declaration-language and serialization invariants (what `Param`,
  `Data`, `Observed`, and `PartiallyObserved` mean; phase boundaries between
  syntax capture, resolved metadata, and serialized IR; wire-format
  guarantees) are stated in bayeswire's `docs/invariants.md` and
  `spec/ir-format-v1.md`, the single normative copies.
- Importing jaxstanv5's backend modules is the only thing that may import
  JAX or BlackJAX. bayeswire must never import them; its no-JAX walk
  enforces that upstream.
- `bind_model(model_cls, values)` is the explicit transition from a
  bayeswire model class to a `BoundModel`, and the only binding path. It
  accepts classes decorated by `@model` or reconstructed with
  `bindable_from_meta(...)`, reading metadata through bayeswire's public
  hooks (`model_meta`, `attached_model_dimensions`) only.
- `BoundModel` is downstream runtime state after concrete data binding. It is
  not part of the authoring/IR boundary and contains no inference logic. It may
  carry validated dimension metadata needed by runtime/export adapters.
- jaxstanv5 proves consume-conformance against the bayeswire corpus read
  from the installed package: every corpus fixture decodes, binds, and
  evaluates within the tolerance policy stated in the bayeswire spec.
- The logp/grad values in the corpus fixtures are produced by this backend
  as the JAX oracle (`scripts/generate_ir_fixtures.py`), in float64, against
  a bayeswire checkout.

## Log density

- The compiler evaluates symbolic distribution arguments before calling the JAX
  backend log-probability operation for each stochastic site.
- Log density = constraint Jacobians + all stochastic site log-density terms.
- The flat unconstrained parameter vector packs free values in the insertion
  order of `free_values`, or of `params` when `free_values` is empty.
- `Param`, `Observed`, and `PartiallyObserved` lower to stochastic sites plus a
  free/fixed coordinate partition; they are not separate log-density machinery.
- Discrete distributions are observed-likelihood distributions only; NUTS
  parameters are continuous and discrete latent `Param(...)` or
  `PartiallyObserved(...)` declarations are rejected.
- Constraints are transform/support metadata, not implicit prior truncation
  syntax. Constrained priors whose base support is wider than the constraint must
  use an explicit `Truncated(...)` distribution with matching concrete bounds.
- `Truncated(...)` is a single explicit wrapper; nested truncation wrappers are
  rejected, and multiple bounds must be flattened into one declaration.
- `MultivariateNormal.scale_tril` must be a valid lower-triangular Cholesky factor
  at bind time. Arbitrary parameter-dependent scale matrices are unsupported;
  a validated Cholesky factor may be multiplied or divided by a provably positive
  scalar expression.
- Sampling returns constrained parameter values.
- Public sampler count arguments (`num_chains`, `num_warmup`, and `num_samples`)
  must be at least 1 and are validated before backend execution.
- Sampling result arrays have shape `(num_chains, num_samples, *param_shape)`,
  including declared zero-sized free values.
- `rhat(...)` and `ess(...)` omit zero-sized sample arrays because they have no
  scalar coordinates to reduce.
- `rhat(...)` reports the maximum classic split-R-hat per non-empty parameter.
- `ess(...)` reports the minimum split-chain Geyer effective sample size per
  non-empty parameter.
- Split-chain diagnostics require at least four post-warmup draws per chain.
- NUTS diagnostics are recorded separately for warmup and post-warmup sampling.
- Diagnostic arrays have shape `(num_chains, num_steps)`, where `num_steps` is
  `num_warmup` for warmup diagnostics and `num_samples` for sampling diagnostics.
- InferenceData compatibility is a typed schema adapter over `BoundModel` and
  `SamplerResult`; it must not import ArviZ, construct xarray datasets, write
  netCDF/zarr artifacts, or change the public sampling result type.

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
