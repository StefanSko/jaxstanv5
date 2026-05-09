# API and Architecture Checklist

Use this checklist before adding or revising public APIs, internal boundaries, or major abstractions.

## Scope fit

- Does this directly help users define a model, express a hierarchical model, compile a model, run NUTS, or inspect NUTS results safely?
- If not, does it belong in `jaxstanv4` core at all?
- Does it conflict with `v4spec.md` non-goals?

## Public vs internal

- Is this concept truly user-facing?
- Can this remain internal?
- Does this expose backend details, benchmark details, or implementation-specific state?
- Would changing the backend later break this API?

## State and invariants

- What invariant does this code protect?
- Is that invariant explicit in the structure or only implied by comments?
- Are there hidden mode combinations that should be separated?
- Can invalid combinations be made unrepresentable or at least harder to construct?

## Surface area

- Is this the smallest API that solves the problem?
- Does it add a second way to do the same thing?
- Does it require extra wrappers, helper layers, or flags to feel usable?
- Can the design be simplified by deleting an option or special case?

## Naming

- Are names direct and responsibility-aligned?
- Would a new contributor understand what the module/type/function does from the name alone?
- Is a generic name hiding avoidable complexity?

## Data flow

- Is data flow explicit from model -> bound -> compiled -> sampled result?
- Are transformations easy to trace?
- Are side effects and mutation minimized?
- Does JAX-traced code stay pure?

## Typing

- Would a small typed container be clearer than a dict or tuple here?
- Is this relying on duck typing or incidental attributes where an explicit private type/protocol would clarify intent?
- Does loose Python input stop at a public boundary, or does it leak into core internals as `Any`?
- Are hierarchical-model states such as symbolic distribution args and data-dependent shapes represented explicitly?
- Are return values structured enough to support safe downstream use?
- Are there optional fields that signal muddled state?

## Dependency discipline

- Does this add a runtime dependency?
- Is that dependency core to the product, or only convenience?
- Can the same goal be achieved with existing runtime dependencies and better code organization?

## Validation impact

- How will this be unit tested?
- How will this be tested against `mcmc-ref` if relevant?
- Does this change require running a targeted reference subset?
- Would repeated-seed robustness on sentinel models be affected?

## Astral validation loop

At minimum before push, and preferably after each meaningful commit, run:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check src tests
uv run pytest
```

If the change touches compiler logic, transforms, binding logic, or the NUTS adapter, also run the relevant reference subset.

## Final test

Can you explain the change in a few lines without mentioning internal incidental machinery?
If not, the design may still be too complicated.
