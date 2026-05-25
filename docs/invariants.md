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
- A model has one or more `Observed` nodes.
- Declaration aliases are invalid: one declaration object maps to one class
  attribute name.

## Phase boundaries

- Class-body syntax capture and resolved model metadata are separate phases.
- Class-body arithmetic creates private deferred syntax, never final expression
  IR.
- `_resolve_model_declaration(...)` is the only transition from declaration
  symbols to named references.
- `bind(...)` is the transition from resolved model class to `BoundModel`.

## IR and module boundaries

- `_deferred.py` is private class-body syntax capture.
- `core.py` does not construct final expression IR.
- `expr.py` is resolved/final IR only.
- Final expression trees contain no declaration symbols, raw declarations, or
  deferred syntax tokens.
- `ModelMeta` contains resolved metadata only.
- `BoundModel` contains no inference logic.

## Log density

- The compiler evaluates symbolic distribution arguments before calling
  `Distribution.log_prob(...)`.
- Log density = constraint Jacobians + parameter priors + all observed
  likelihood terms.
- Sampling returns constrained parameter values.
