# AGENTS.md

## Project identity

`jaxstanv5` is the **JAX/BlackJAX sampling backend for
[bayeswire](https://github.com/StefanSko/bayeswire) models**.

It exists to help users:
- bind concrete data to bayeswire model declarations (`bind_model`)
- compile models to executable log densities
- handle constraints and Jacobian-aware transforms
- run **NUTS only**
- inspect essential diagnostics (`rhat`, `ess`, divergences)
- simulate priors/prior-predictive and export the InferenceData-compatible schema

The model declaration language, distributions/constraints metadata, IR wire
format, spec, and conformance corpus live in bayeswire, pinned by exact
version; the pin-bump diff is the compatibility review.

It does **not** exist to be a workflow platform, reporting system, plotting toolkit,
multi-algorithm inference playground, session/artifact product, viewer, or a
second home for the declaration language.

## Communication

Be precise and brief. Do not pad responses. State assumptions, uncertainty, and tradeoffs explicitly.

## File handling

Read full relevant files before editing. Do not make changes from partial context unless the file is trivial. Prefer small, targeted edits.

## Typing

Use strict typing throughout. Never use `Any`. Avoid untyped structured dictionaries; prefer dataclasses, `TypedDict`, enums, protocols, and explicit result types.

## Architecture

Follow `.pi/skills/rust-style-python` and keep the project invariants in
[`docs/invariants.md`](docs/invariants.md) true.

Prefer:
- immutable or effectively immutable data
- narrow public APIs
- explicit state transitions
- typed return values
- small modules with clear responsibilities
- designs that make invalid states hard to represent

Avoid:
- hidden mutation
- hidden control flow
- duck typing in core semantics
- speculative abstractions
- broad convenience APIs

### Phase boundaries

When a design has distinct phases, make those phases explicit in code. Use
separate types and named transition functions instead of clever mixed
representations or hidden conversions.

Avoid shortcuts that make the code look simpler by increasing coupling or hiding
state transitions. Only collapse phases for a clear performance reason, and
document that tradeoff.

## Development process

Work collaboratively with the user. Do not treat the project instructions as an
autonomous end-to-end workflow. Wait for user confirmation when scope,
architecture, test strategy, or cleanup direction is unclear.

Use staged red-green TDD for user-visible additions, but apply it from the
current agreed state; not every task starts at step 1.

Typical flow:

1. If no public workflow target exists yet, add or update a red integration test
   for the intended behavior. For new distributions or other conceptual modeling
   surfaces, the declaration metadata lands in bayeswire first (with a corpus
   reference model there); the backend work starts with a north-star fixture in
   `tests/integration/_reference_models.py` and coverage in
   `tests/integration/test_distribution_coverage.py`.
2. If the integration test is too broad or noisy, add a focused red seam test for
   the first architectural path that should work.
3. Drive the path with red/green unit tests for each primitive.
4. Make the focused path green, then make the integration path green or document
   why it remains intentionally red.
5. Keep focused seam tests only when they remain useful durable coverage;
   otherwise merge or delete them after the integration path is green.
6. Refactor only after the relevant tests are green.

Prefer vertical integration and iterative development. Work toward one
end-to-end path first, then improve breadth and polish.

Test public behavior through public APIs; do not interrogate objects in ways that
leak private internals into higher-level tests. Avoid monkeypatching and mocking;
prefer real collaborators and behavioral tests.

When adding a wrapper type, audit all relevant `isinstance` dispatch sites up
front. For numerical code, write adversarial tail and gradient tests before the
implementation.

## Validation

Before reporting completion, run the relevant checks, normally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Briefly report the result.
