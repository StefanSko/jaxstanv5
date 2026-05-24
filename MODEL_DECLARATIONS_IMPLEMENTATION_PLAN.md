# Detailed Plan: Model Declarations and `@model`

This document tracks the model-declaration implementation after the design
simplification that removed the internal pending IR.

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

The implementation has a two-step declaration path:

```text
Python class body syntax
  -> resolved ModelMeta
  -> BoundModel
```

Concrete files:

```text
src/jaxstanv5/model/_deferred.py  # private class-body syntax tokens
src/jaxstanv5/model/expr.py       # final/resolved expression IR only
src/jaxstanv5/model/core.py       # Param, Data, Observed declarations
src/jaxstanv5/model/decorator.py  # resolve declaration -> attach metadata/bind
src/jaxstanv5/model/bound.py      # BoundModel
```

Important invariants:

- `expr.py` is final/resolved only: `ParamRef.name: str`, `DataRef.name: str`.
- `core.py` never constructs final `expr.py` nodes.
- Class-body operators create private deferred syntax tokens, not semantic IR.
- `_resolve_model_declaration(...)` is the only transition from declaration
  symbols to final named refs.

This avoids a mixed intermediate representation such as
`BinOp(ParamRef(DeclarationSymbol(...)), ...)`.

---

## Phase model

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
- `mu` is a private deferred syntax token.
- declarations carry private `DeclarationSymbol` identities.

Invariant:

```text
class-body arithmetic captures syntax only; it must not produce ExprNode values
```

### 2. Private deferred syntax capture

File: `src/jaxstanv5/model/_deferred.py`

Types:

```python
DeclarationSymbol
DeferredBinOp
DeferredIndexOp
DeferredExpr = DeferredBinOp | DeferredIndexOp
```

Deferred tokens are not semantic model IR. They are only the raw Python syntax
that was evaluated before class attribute names were available.

Example:

```python
alpha + beta * x
```

captures:

```python
DeferredBinOp(
    "+",
    alpha,
    DeferredBinOp("*", beta, x),
)
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
final ExprNode trees never contain DeclarationSymbol, Param, Data, or deferred tokens
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
- declaration operators (`+`, `-`, `*`, `/`, reverse versions, `[]`) capture
  private deferred syntax tokens.
- unsupported expression operands are rejected during declaration resolution,
  not during Python operator evaluation.

---

## Decorator resolution

File: `src/jaxstanv5/model/decorator.py`

The decorator should remain a small explicit transition:

```python
def model(cls: type[object]) -> type[object]:
    meta = _resolve_model_declaration(cls)
    attach metadata and bind
    return cls
```

### `_resolve_model_declaration(cls)`

Responsibilities:

1. Walk `cls.__dict__` and collect declaration symbols.
2. Reject aliases such as `beta = alpha` where two names share one declaration.
3. Collect parameters and explicit data slots.
4. Resolve parameter distributions directly to final expression nodes where
   symbolic fields appear.
5. Resolve parameter sizes:
   - `None -> None`
   - `int -> int`
   - `Data -> DataRef(name)`
6. Collect exactly one `Observed` declaration.
7. Resolve deferred class-body expression attributes to final `ExprNode` trees.
8. Return final `ModelMeta`.

Distribution resolution examples:

```python
Normal(alpha_pop, sigma_alpha)
```

becomes final metadata equivalent to:

```python
Normal(ParamRef("alpha_pop"), ParamRef("sigma_alpha"))
```

```python
Observed(Normal(mu, sigma))
```

where `mu` is deferred syntax and `sigma` is a `Param`, becomes:

```python
Observed(Normal(<final mu tree>, ParamRef("sigma")))
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
- no declaration symbols remain.
- no raw `Param`, `Data`, or `Observed` declarations remain.
- no deferred tokens remain.

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

Do not export `_deferred.py` from public package surfaces.

---

## Current status

Implemented:

- private deferred syntax capture in `_deferred.py`
- final expression IR in `expr.py`
- declaration classes in `core.py`
- explicit `_resolve_model_declaration -> model` pipeline
- distribution and size resolution for hierarchical declarations
- `BoundModel`
- `bind(...)`
- public DSL exports
- unit coverage for deferred syntax capture, declaration resolution, validation,
  and binding edge cases
- slice coverage for public `@model` declarations and bind behavior

Focused validation should pass with:

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

1. Decide whether scalar distribution fields should remain raw scalars or be
   normalized to expression constants. Current implementation normalizes scalar
   distribution fields to final constant nodes; this is uniform for compiler
   evaluation, but direct `Distribution.log_prob(...)` on unresolved metadata is
   not expected to work.
2. Tighten the type of attached `bind(...)` if it becomes useful for static
   checks. Runtime behavior is currently the priority.
3. Introduce explicit symbolic distribution parameter types or model-side
   distribution metadata if casts around `DistributionParameter` become too
   noisy during compiler work.

---

## Design invariants

Keep these true:

- `Param` and `Data` are declarations.
- `_deferred.py` is private and owns class-body syntax capture.
- `core.py` does not import or build final expression IR.
- `expr.py` is resolved/final only.
- `ParamRef.name` and `DataRef.name` are strings.
- `@model`/`_resolve_model_declaration` is the only transition from declaration
  symbols to names.
- `Observed` is not a `Data` slot, but `bind(...)` still requires observed data.
- `BoundModel` contains data and shapes only; no inference logic.
- no BlackJAX in model code.
- no compiler logic in model code.
- prefer immutable dataclasses.
- do not use `Any`.
