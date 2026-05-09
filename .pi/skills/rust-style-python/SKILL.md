---
name: rust-style-python
description: Rust-inspired Python architecture and API design rules for immutable data, explicit state transitions, narrow public interfaces, predictable module boundaries, and type-driven refactors. Use when designing jaxstanv4 APIs, refactoring internals, reviewing abstractions, or deciding whether code belongs in core.
---

# Rust Style Python

Use this skill for architecture work in `jaxstanv4`, especially when:
- designing or revising public APIs
- refactoring internal boundaries
- deciding whether an abstraction belongs in core
- reviewing state handling, mutability, and type surfaces
- simplifying modules without weakening correctness

This skill is meant to preserve the design discipline used in earlier jaxstan work while fitting the smaller v4 scope.

## Primary goal

Make the codebase:
- smaller
- clearer
- harder to misuse
- easier to validate
- more explicit about state and invariants

Favor designs that make invalid states hard to represent and public behavior easy to explain.

## Core rules

1. Prefer immutable or effectively immutable data structures.
2. Keep public APIs narrow and boring.
3. Make state transitions explicit.
4. Separate public contracts from backend/internal machinery.
5. Collapse abstractions that do not protect correctness or clarity.
6. Use typed return values instead of ad hoc dicts where practical.
7. Keep modules organized around responsibilities, not convenience exports.
8. Avoid hidden mutation, hidden caching, and hidden control flow.
9. Prefer a small number of obvious paths over a wide convenience surface.
10. Validate before generalizing.
11. Keep loose Python input at public boundaries and normalize quickly into explicit internal types.
12. Avoid duck typing for core semantics unless it is deliberately isolated and tested at an interoperability boundary.

## Project-specific rules for jaxstanv4

- The public story is: define model -> bind data -> sample with NUTS.
- BlackJAX must remain an internal detail.
- `mcmc-ref` must remain a dev/test validation dependency, not a runtime concept.
- Do not reintroduce workflow/reporting/product layers into core.
- Keep all distributions, but keep architecture minimal.
- Treat hierarchical models as first-class; symbolic distribution parameters are core declaration metadata, not one-off exceptions.
- Prefer explicit private dataclasses/tagged states over incidental attributes or duck-typed hooks for model, shape, distribution, compiler, and result internals.
- If a new abstraction exists mainly to support a possible future feature, remove or defer it.

## Required reading for non-trivial design work

Before making a substantial architecture or API change, read:
- `references/principles.md`
- `references/api-checklist.md`
- `../../../v4spec.md`
- `../../../AGENTS.md`

## Working method

1. Identify the real invariant or boundary the code needs to enforce.
2. Check whether the current design exposes backend details or unnecessary surface area.
3. Rewrite toward smaller explicit data types and narrower interfaces.
4. Ensure naming matches responsibility.
5. Run the Astral validation loop and any relevant reference subset.
6. Re-check whether the resulting design is easier to explain in one paragraph.

## Red flags

Stop and reconsider if a change introduces:
- backend types in public APIs
- generic managers/factories/registries without strong justification
- boolean mode flags that imply hidden state machines
- runtime-only checks for states that could be represented in types/structure
- duck typing or `Any`-heavy internals where a small explicit private type would clarify the invariant
- convenience wrappers that duplicate the main happy path
- abstractions created only for hypothetical future inference algorithms

## Output expectations

When using this skill in a design or review task:
- identify the invariant
- state what should be public vs internal
- point out needless abstraction
- propose the smallest viable interface
- explain how the result improves correctness, clarity, or maintainability
