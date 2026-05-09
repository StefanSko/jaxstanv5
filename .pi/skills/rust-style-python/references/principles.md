# Rust-Style Python Principles

These principles are intended for `jaxstanv4`: a small library with a narrow public story and strong correctness constraints.

## 1. Make invalid states hard to represent

Prefer structures that encode what is valid by construction.

Examples:
- keep incomplete vs bound vs compiled concepts separate
- distinguish public result types from backend sampler state
- avoid optional fields that are only valid in some undocumented modes

If two states have different invariants, they probably deserve different types or at least different explicit representations.

For `jaxstanv4`, this includes symbolic-vs-numeric distribution arguments,
static-vs-data-dependent shape dimensions, declared-vs-bound model state, and
backend-neutral-vs-BlackJAX sampler state.

## 2. Keep the happy path obvious

A user should be able to understand the library through one path:

```python
@model
class MyModel:
    ...

bound = MyModel.bind(...)
result = sample(bound, key=...)
```

Anything that weakens that story should be justified carefully.

Hierarchical models are part of this happy path, not an advanced side channel.
The user should be able to write hierarchical declarations with ordinary model
expressions as distribution arguments and rely on the compiler to resolve them
later through explicit internal states.

## 3. Prefer narrow interfaces over broad convenience

Good abstractions remove accidental complexity.
Bad abstractions hide the real behavior and add more names than value.

Ask:
- does this API reduce concepts?
- or does it just spread one concept across more files and wrappers?

## 4. Separate contract from mechanism

Public API should expose stable concepts.
Internal code may change freely.

For `jaxstanv4`:
- public: models, distributions, constraints, compile, sample, results, diagnostics
- internal: BlackJAX adapter, warmup logic, packing/layout details, benchmark harness details

## 5. Prefer explicit state transitions

Instead of hidden mode changes, prefer explicit phases:
- declared model
- bound model
- compiled model
- sampled result

Even when not formalized in the type system, code structure should make these transitions obvious.

## 6. Avoid stringly, dictly, or duck-typed architecture where a type would clarify intent

Small dataclasses or typed containers are often better than ad hoc nested dicts.
Use dicts when the domain is naturally map-shaped; do not default to them for convenience.

Do not rely on incidental methods or attributes to identify core internal states
when an explicit private type or protocol would make the invariant clearer. Loose
Python objects are fine at public boundaries; normalize them before they spread.

## 7. Localize complexity

Some complexity is real, especially around transforms, packing, and NUTS integration. Keep that complexity behind narrow internal seams rather than leaking it across the codebase.

## 8. Delete speculative architecture

Do not keep generic layers around for features explicitly out of scope for v4.
If the only reason an abstraction exists is “maybe we add five inference engines later,” it probably should not exist now.

## 9. Favor boring names and direct modules

Prefer:
- `sample`
- `compile_model`
- `SampleResult`
- `nuts_backend`

Avoid abstract names that obscure responsibility unless they buy real precision.

## 10. Validation is part of design

A good design is easier to validate.
If a design makes it hard to test transforms, compiled densities, or reference posterior behavior, the design may be wrong.

## 11. Minimize cross-cutting magic

Especially in a DSL, hidden behavior can become expensive quickly.
Keep the implementation understandable:
- where fields are discovered
- how expressions are compiled
- how constraints are applied
- how unconstrained state is mapped back to named draws

## 12. Optimize for maintainability under scientific pressure

This project is not just software; it is inference software.
Choose designs that make correctness reviews, benchmark analysis, and regression debugging easier.
