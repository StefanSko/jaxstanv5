# Test layout

Tests are grouped by purpose:

```text
tests/unit/         small tests for isolated primitives and internal transitions
tests/integration/  public workflows and cross-module behavior
```

Use `unit` for local correctness. Use `integration` when behavior depends on the
public model declaration, binding, compiler, inference, or diagnostics workflow.
