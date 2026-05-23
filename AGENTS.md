# AGENTS.md

## Project identity

`jaxstanv5` is a **minimal declarative Bayesian modeling library for JAX**.

It exists to help users:
- define models with `@model`, `Param`, `Data`, and `Observed`
- express hierarchical models as first-class model declarations
- compile models to executable log densities
- handle constraints and Jacobian-aware transforms
- run **NUTS only**
- inspect essential diagnostics (`rhat`, `ess`, divergences)

It does **not** exist to be a workflow platform, reporting system, plotting toolkit,
multi-algorithm inference playground, session/artifact product, or viewer.

## Communication

Be precise and brief. Do not pad responses. State assumptions, uncertainty, and tradeoffs explicitly.

## File handling

Read full relevant files before editing. Do not make changes from partial context unless the file is trivial. Prefer small, targeted edits.

## Typing

Use strict typing throughout. Never use `Any`. Avoid untyped structured dictionaries; prefer dataclasses, `TypedDict`, enums, protocols, and explicit result types.

## Architecture

Follow `.pi/skills/rust-style-python`.

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

Use staged red-green TDD for user-visible additions:
1. Add or update a red integration test for the intended public workflow.
2. Add a red slice test for the first architectural path that should work.
3. Drive the slice with red/green unit tests for each primitive.
4. Make the slice green, then make the integration path green or document why it remains intentionally red.
5. Refactor only after the relevant tests are green.

Prefer vertical integration and iterative development. Work toward one end-to-end path first, then improve breadth and polish.

Test public behavior through public APIs; do not interrogate objects in ways that leak private internals into higher-level tests.
Avoid monkeypatching and mocking; prefer real collaborators and behavioral tests.

## Validation

Before reporting completion, run the relevant checks, normally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Briefly report the result.
