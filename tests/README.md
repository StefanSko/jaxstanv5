# Test layout

Tests are grouped by purpose:

```text
tests/unit/         small tests for isolated primitives
tests/slices/       focused feature slices that cross module boundaries
tests/integration/  user-level workflows across subsystems
```

Use `unit` for local correctness, `slices` for architectural pipelines such as
model declaration phases, and `integration` for end-to-end product behavior.
