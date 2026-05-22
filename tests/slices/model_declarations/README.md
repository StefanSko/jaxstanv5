# Model declaration slice tests

This is a feature-slice test suite. It verifies the model declaration path from
real `@model` declarations to final metadata and `BoundModel` binding behavior.

```text
@model class declaration
  -> final ModelMeta
  -> bind(...) / BoundModel
```

These tests intentionally avoid the private pending IR. Pending expression and
resolution details are covered by unit tests in `tests/unit/model/`.

Related primitive tests live in `tests/unit/model/`:

- `test_expr_ir.py` checks final/resolved expression nodes.
- `test_pending_expr_ir.py` checks class-body pending expression nodes.
- `test_decorator_resolution.py` checks pending-to-final resolution helpers.
- `test_decorator_validation.py` checks declaration validation.
- `test_bind.py` checks binding and parameter shape resolution.
