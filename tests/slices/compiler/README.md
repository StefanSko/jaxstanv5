# Compiler slice tests

Feature-slice tests for the compiler pipeline:

```text
BoundModel → compile_log_density → callable log_prob(params) → scalar
```

These tests use real `@model` declarations and verify that compiled log-density
functions return correct numerical values against hand-computed references.

Related primitive tests live in `tests/unit/compiler/`:

- `test_evaluate_expr.py` — ExprNode tree → jax.Array evaluation
- `test_evaluate_distribution.py` — Distribution field resolution
- `test_compile_pipeline.py` — full compile → log_prob numerical correctness
