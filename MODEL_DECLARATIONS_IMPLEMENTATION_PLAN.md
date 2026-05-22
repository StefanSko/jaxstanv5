# Detailed Plan: Model Declarations and `@model`

This document tracks the model-declaration implementation after the design change
that made the class-body phase explicit.

The supported DSL remains:

```python
@model
class LinearRegression:
    alpha = Param(Normal(0.0, 1.0))
    beta = Param(Normal(0.0, 1.0))
    sigma = Param(Normal(0.0, 1.0), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
```

And hierarchical parameters:

```python
@model
class HierarchicalRegression:
    n_groups = Data()
    alpha_pop = Param(Normal(0.0, 1.0))
    sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    y = Observed(Normal(alpha, 1.0))
```

---

## Current architecture

The implementation now has an explicit phase pipeline:

```text
Python class body
  -> pending expression IR / PendingModel
  -> resolved ModelMeta
  -> BoundModel
```

Concrete files:

```text
src/jaxstanv5/model/_pending.py   # internal class-body / unresolved IR
src/jaxstanv5/model/expr.py       # final/resolved expression IR only
src/jaxstanv5/model/core.py       # Param, Data, Observed declarations
src/jaxstanv5/model/decorator.py  # collect -> resolve -> attach
src/jaxstanv5/model/bound.py      # BoundModel
```

Important design change from the original plan:

- `UnresolvedSymbol` is **not** in `expr.py` anymore.
- `expr.py` is final/resolved only: `ParamRef.name: str`, `DataRef.name: str`.
- Class-body expressions use internal pending nodes from `_pending.py`.
- The decorator is the only transition from pending symbols to final names.

This avoids a mixed intermediate representation such as
`BinOp(ParamRef(UnresolvedSymbol(...)), ...)`.

---

## Phase model and invariants

### 1. Class-body declaration phase

During class body execution, declarations do not know their final attribute names.

```python
alpha = Param(...)
x = Data()
mu = alpha + x
```

At this point:

- `alpha` is a `Param` declaration.
- `x` is a `Data` declaration.
- `mu` is a pending expression tree.
- declarations carry `UnresolvedSymbol` values from `_pending.py`.

Invariant:

```text
class-body arithmetic must produce PendingExprNode values only
```

No final `expr.py` nodes should appear in pending expression trees.

### 2. Internal pending IR

File: `src/jaxstanv5/model/_pending.py`

Types:

```python
UnresolvedSymbol
PendingParamRef
PendingDataRef
PendingConst
PendingBinOp
PendingIndexOp
PendingRef = PendingParamRef | PendingDataRef
PendingExprNode = PendingRef | PendingConst | PendingBinOp | PendingIndexOp
```

`_pending.py` is package-private. It exists only to make the declaration phase
explicit in the type system.

Pending refs use unresolved symbols:

```python
PendingParamRef(UnresolvedSymbol(0))
PendingDataRef(UnresolvedSymbol(1))
```

### 3. Final expression IR

File: `src/jaxstanv5/model/expr.py`

Types:

```python
ParamRef      # name: str
DataRef       # name: str
ConstNode
BinOp
IndexOp
ExprNode = ParamRef | DataRef | ConstNode | BinOp | IndexOp
```

Invariant:

```text
final ExprNode trees never contain UnresolvedSymbol or pending nodes
```

Example final tree:

```python
BinOp("+", ParamRef("alpha"), DataRef("x"))
```

---

## Declaration classes

File: `src/jaxstanv5/model/core.py`

Declaration types:

```python
Param
Data
Observed
```

Rules:

- `Param` and `Data` are declarations, not expression nodes.
- `Observed` is an observed-variable declaration, not a `Data` slot.
- `Param.ref()` returns `PendingParamRef`.
- `Data.ref()` returns `PendingDataRef`.
- declaration operators (`+`, `-`, `*`, `/`, reverse versions, `[]`) build pending nodes.

Example:

```python
alpha + beta * x
```

must build:

```python
PendingBinOp(
    "+",
    PendingParamRef(alpha.symbol),
    PendingBinOp("*", PendingParamRef(beta.symbol), PendingDataRef(x.symbol)),
)
```

---

## Decorator pipeline

File: `src/jaxstanv5/model/decorator.py`

The decorator should remain a small pipeline:

```python
def model(cls: type[object]) -> type[object]:
    pending = collect_pending_model(cls)
    meta = resolve_pending_model(pending)
    attach metadata and bind
    return cls
```

### `PendingModel`

Collected before symbol resolution:

```python
@dataclass(frozen=True)
class PendingModel:
    params: dict[str, PendingParam]
    data_slots: list[str]
    observed_name: str
    observed: PendingObserved
    expressions: dict[str, PendingExprNode]
    symbols: dict[UnresolvedSymbol, str]
```

Pending metadata invariants:

- `symbols` maps every declaration symbol to the final class attribute name.
- `data_slots` includes only explicit `Data()` declarations.
- `observed_name` is separate from `data_slots`.
- pending expressions contain only `_pending.py` nodes.
- pending metadata should not embed raw `Param`, `Data`, or `Observed` declarations.

### `collect_pending_model(cls)`

Responsibilities:

1. Walk `cls.__dict__` and collect declaration symbols.
2. Collect pending parameters and explicit data slots.
3. Normalize parameter distributions to pending references where needed.
4. Normalize parameter sizes:
   - `None -> None`
   - `int -> int`
   - `Data -> PendingDataRef`
   - `PendingDataRef -> PendingDataRef`
5. Collect exactly one `Observed` declaration.
6. Collect class-body pending expression attributes.

Distribution normalization examples:

```python
Normal(alpha_pop, sigma_alpha)
```

becomes pending metadata equivalent to:

```python
Normal(PendingParamRef(alpha_pop.symbol), PendingParamRef(sigma_alpha.symbol))
```

```python
Observed(Normal(mu, sigma))
```

where `mu` is a pending expression and `sigma` is a `Param`, becomes:

```python
Observed(Normal(<pending mu tree>, PendingParamRef(sigma.symbol)))
```

### `ModelMeta`

Resolved final metadata:

```python
@dataclass(frozen=True)
class ModelMeta:
    params: dict[str, ResolvedParam]
    data_slots: list[str]
    observed_name: str
    observed: ResolvedObserved
    expressions: dict[str, ExprNode]
```

Final metadata invariants:

- final expressions contain only `expr.py` nodes.
- `ParamRef.name` and `DataRef.name` are strings.
- no pending nodes remain.
- no unresolved symbols remain.
- no raw `Param`, `Data`, or `Observed` declarations remain.

### `resolve_pending_model(pending)`

Responsibilities:

1. Resolve `PendingParamRef(symbol)` to `ParamRef(name)`.
2. Resolve `PendingDataRef(symbol)` to `DataRef(name)`.
3. Resolve `PendingConst` to `ConstNode`.
4. Resolve `PendingBinOp` and `PendingIndexOp` recursively.
5. Resolve pending distribution fields.
6. Resolve parameter sizes:
   - `None -> None`
   - `int -> int`
   - `PendingDataRef(symbol) -> DataRef(name)`

Example:

```python
PendingBinOp("+", PendingParamRef(sym_alpha), PendingDataRef(sym_x))
```

becomes:

```python
BinOp("+", ParamRef("alpha"), DataRef("x"))
```

---

## Binding

File: `src/jaxstanv5/model/bound.py`

```python
@dataclass(frozen=True)
class BoundModel:
    meta: ModelMeta
    data: dict[str, jax.Array]
    param_shapes: dict[str, tuple[int, ...]]
    n_params: int
```

`bind(...)` is attached by `@model` and performs the explicit transition:

```text
resolved model class -> BoundModel
```

Responsibilities:

1. Require all explicit data slots.
2. Require observed data by `observed_name`.
3. Reject extra data.
4. Convert values to JAX arrays.
5. Resolve parameter shapes.
6. Compute total constrained parameter count.

Shape rules:

```python
None       -> ()
int        -> (size,)
DataRef(n) -> (int(bound_data[n]),)
```

Parameter count:

```python
()      -> 1
(3,)    -> 3
(2, 4)  -> 8  # future; current slice needs scalar and 1D sizes only
```

---

## Public exports

File: `src/jaxstanv5/model/__init__.py`

```python
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import ModelMeta, model

__all__ = ["BoundModel", "Data", "ModelMeta", "Observed", "Param", "model"]
```

File: `src/jaxstanv5/__init__.py`

```python
from jaxstanv5.model import Data, Observed, Param, model

__all__ = ["Data", "Observed", "Param", "model"]
```

Do not export `_pending.py` from public package surfaces.

---

## Current status

Implemented:

- red model declaration tests
- internal pending IR in `_pending.py`
- final expression IR in `expr.py`
- declaration classes in `core.py`
- explicit `collect_pending_model -> resolve_pending_model -> model` pipeline
- distribution and size normalization for hierarchical declarations
- `BoundModel`
- `bind(...)`
- public DSL exports

Focused validation passed:

```bash
uv run ruff check src/jaxstanv5 tests/unit tests/slices/model_declarations
uv run ty check src/jaxstanv5 tests/unit tests/slices/model_declarations
uv run pytest \
  tests/unit \
  tests/slices/model_declarations \
  -q
```

Full-project pytest is not expected to pass yet because later vertical-slice APIs
such as diagnostics exports are still unimplemented.

---

## Known follow-ups

These are not blockers for the current model-declaration slice, but should be
revisited before compiler work expands:

1. Add direct tests for missing observed declaration and duplicate observed
   declarations.
2. Add direct tests for `bind(...)` missing data, missing observed data, and extra
   data rejection.
3. Decide whether scalar distribution fields should remain raw scalars or be
   normalized to expression constants. Current implementation normalizes scalar
   distribution fields through pending/final constant nodes; this is uniform for
   compiler evaluation, but direct `Distribution.log_prob(...)` on unresolved
   metadata is not expected to work.
4. Tighten the type of attached `bind(...)` if it becomes useful for static
   checks. Runtime behavior is currently the priority.

---

## Design invariants

Keep these true:

- `Param` and `Data` are declarations.
- `_pending.py` is internal and owns `UnresolvedSymbol`.
- class-body expressions produce pending nodes only.
- `expr.py` is resolved/final only.
- `ParamRef.name` and `DataRef.name` are strings.
- `@model` is the only transition from unresolved symbols to names.
- `Observed` is not a `Data` slot, but `bind(...)` still requires observed data.
- `BoundModel` contains data and shapes only; no inference logic.
- no BlackJAX in model code.
- no compiler logic in model code.
- prefer immutable dataclasses.
- do not use `Any`.
