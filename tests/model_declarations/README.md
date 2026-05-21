# Model declaration tests

These tests document the explicit model-declaration pipeline:

```text
final expression IR primitives
  -> class-body pending expression IR
  -> collect_pending_model(...)
  -> resolve_pending_model(...)
  -> @model public DSL
  -> bind(...) / BoundModel
```

Files are numbered in dependency order:

1. `test_01_final_expr_ir.py` checks the resolved expression nodes used after
   model decoration.
2. `test_02_pending_expr_ir.py` checks class-body arithmetic produces internal
   pending nodes, not final expression nodes.
3. `test_03_phase_pipeline.py` checks the explicit pending-to-final phase
   transitions and invariants.
4. `test_04_public_dsl_bind.py` checks the user-facing `@model` and `bind(...)`
   path.

The internal pending IR lives in `jaxstanv5.model._pending`. It is intentionally
not public API; it is tested here because it defines the declaration phase
boundary.
