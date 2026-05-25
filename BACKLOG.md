# Backlog

Small, concrete improvement ideas that are not required for the current working vertical slice.

## Model declaration typing

- Introduce explicit symbolic distribution parameter types if casts around
  `DistributionParameter` become too noisy as distributions expand.
- Consider phase-specific distribution metadata if plain distribution dataclasses
  become awkward:
  - declaration-time distribution parameters
  - resolved symbolic distribution parameters
  - executable numeric distribution parameters

## Decorated model typing

- Improve static typing for classes returned by `@model`.
  Runtime behavior is fine, but tests currently need protocols/casts because
  `_model_meta` and `bind(...)` are attached dynamically.

## Distribution normalization

- Decide whether scalar distribution fields should remain raw scalars or always
  normalize to `ConstNode` in resolved model metadata.
- Revisit generic dataclass-based distribution normalization as more
  distributions are added.

## Model declaration validation

- Keep expanding validation around declaration edge cases as new DSL features are
  added.
- Decide explicit behavior for unsupported distribution field values before more
  compiler behavior depends on them.

## Diagnostics

- Expose and test divergences as part of the public sampling result once the
  minimal diagnostics API settles.
