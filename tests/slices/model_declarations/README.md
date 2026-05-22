# Model declaration slice tests

This is a feature-slice test suite. It verifies the model declaration pipeline
from class-body declarations to `BoundModel`.

```text
class body pending IR
  -> collect_pending_model(...)
  -> resolve_pending_model(...)
  -> @model public DSL
  -> bind(...) / BoundModel
```

These tests intentionally cross model modules (`core`, `_pending`, `decorator`,
`expr`, and `bound`) but stop before compiler or sampler integration.

Related primitive tests live in `tests/unit/model/`:

- `test_expr_ir.py` checks final/resolved expression nodes.
- `test_pending_expr_ir.py` checks class-body pending expression nodes.
