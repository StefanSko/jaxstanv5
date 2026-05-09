# AGENTS.md

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
