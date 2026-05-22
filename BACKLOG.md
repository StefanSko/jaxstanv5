# Backlog

Small, concrete improvement ideas that are not required for the current model-declaration slice.

## Model declaration typing

- Introduce explicit symbolic distribution parameter types once compiler work starts.
  Current tests need casts because runtime distributions are typed for numeric/JAX
  parameters, while model declarations pass symbolic pending/final expressions.
- Consider phase-specific distribution metadata if plain distribution dataclasses
  become awkward:
  - pending distribution parameters
  - resolved distribution parameters
  - executable numeric distribution parameters

## Decorated model typing

- Improve static typing for classes returned by `@model`.
  Runtime behavior is fine, but tests currently need protocols/casts because
  `_model_meta` and `bind(...)` are attached dynamically.

## Distribution normalization

- Revisit generic dataclass rebuilding in `rebuild_distribution(...)`.
  It is convenient now, but explicit model-side distribution metadata may become
  clearer as more distributions are added.

## Model declaration validation

- Keep expanding validation around declaration edge cases as new DSL features are
  added.
- Decide explicit behavior for unsupported distribution field values before
  compiler work depends on them.

## Integration path

- Keep the red integration test as the public workflow target.
- Implement missing vertical-slice APIs in order:
  1. compile `BoundModel` to log density
  2. expose NUTS-only `sample(...)`
  3. expose diagnostics `rhat`, `ess`, divergences
