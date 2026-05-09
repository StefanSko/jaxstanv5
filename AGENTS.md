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

## Development process

Use red-green TDD:
1. Add or update a failing test.
2. Implement the smallest change to make it pass.
3. Refactor only after the test is green.

Prefer vertical integration and iterative development. When asked to build application/product functionality, work toward an end-to-end finished path first, then improve breadth and polish.

## Validation

Before reporting completion, run the relevant checks, normally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Briefly report the result.
