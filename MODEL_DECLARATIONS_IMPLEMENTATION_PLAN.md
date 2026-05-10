# Detailed Plan: Model Declarations and `@model`

This plan replaces the previous implementation attempt. The main goal is to keep the design easier to follow while still supporting the desired DSL:

```python
@model
class LinearRegression:
    alpha = Param(Normal(0, 1))
    beta = Param(Normal(0, 1))
    sigma = Param(Normal(0, 1), constraint=Positive())
    x = Data()
    mu = alpha + beta * x
    y = Observed(Normal(mu, sigma))
```

And hierarchical parameters:

```python
n_groups = Data()
alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
```

## Core challenge

Inside a Python class body, this code executes immediately:

```python
mu = alpha + beta * x
```

So `Param(...)` and `Data()` must be objects that can participate in arithmetic and produce symbolic expression nodes.

The tricky part is that during class-body execution, declarations do not know their final attribute names yet. For example, `alpha = Param(...)` only becomes named `"alpha"` after the class body has finished.

Therefore, use a simple two-step approach:

1. During class-body execution, declarations create expression references using explicit `UnresolvedSymbol` values.
2. In the `@model` decorator, replace those unresolved symbols with final attribute names.

This is not a monad. It is a deferred symbolic expression graph with a clear terminal operation: `@model` resolves symbols into names.

---

## Phase 4A: Write red tests first

Create `tests/test_model_declarations.py`.

Test three things only:

1. The decorator collects declarations.
2. `bind(...)` attaches concrete data and resolves scalar parameter shapes.
3. `bind(...)` resolves data-dependent hierarchical parameter sizes.

Suggested test model:

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

Expected metadata:

```python
meta = LinearRegression._model_meta
assert list(meta.params) == ["alpha", "beta", "sigma"]
assert meta.data_slots == ["x"]
assert meta.observed_name == "y"
assert set(meta.expressions) == {"mu"}
```

Expected bind result:

```python
bound = LinearRegression.bind(x=jnp.asarray([1.0]), y=jnp.asarray([2.0]))
assert bound.param_shapes == {"alpha": (), "beta": (), "sigma": ()}
assert bound.n_params == 3
```

Hierarchical model:

```python
@model
class HierarchicalRegression:
    n_groups = Data()
    alpha_pop = Param(Normal(0.0, 1.0))
    sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
    alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
    y = Observed(Normal(alpha, 1.0))
```

Expected:

```python
bound = HierarchicalRegression.bind(n_groups=3, y=jnp.asarray([0.0, 1.0, 2.0]))
assert bound.param_shapes["alpha_pop"] == ()
assert bound.param_shapes["sigma_alpha"] == ()
assert bound.param_shapes["alpha"] == (3,)
assert bound.n_params == 5
```

Run the test and confirm it fails on missing `Data`, `Param`, `Observed`, or `model`.

---

## Phase 4B: Keep the expression system as the stable foundation

Existing expression nodes already include:

- `ParamRef`
- `DataRef`
- `ConstNode`
- `BinOp`
- `IndexOp`

Add one explicit symbol type:

```python
@dataclass(frozen=True)
class UnresolvedSymbol:
    id: int
```

Then define:

```python
type Symbol = str | UnresolvedSymbol
```

And allow:

```python
ParamRef.name: Symbol
DataRef.name: Symbol
```

Why?

- Before decoration: references use `UnresolvedSymbol(...)`.
- After decoration: references use real string names like `"alpha"` and `"x"`.

This makes the lifecycle explicit:

```text
class body:  ParamRef(UnresolvedSymbol(0)), DataRef(UnresolvedSymbol(1))
@model:      ParamRef("alpha"), DataRef("x")
```

Avoid bare integers for unresolved names. Bare `int` works technically, but `UnresolvedSymbol` communicates intent and prevents accidental confusion with ordinary numeric constants.

Define `UnresolvedSymbol` in `model/expr.py`, because expression references need to carry it.

Because `UnresolvedSymbol` is a frozen dataclass, it is hashable and can be used as a key in the decorator symbol table.

Avoid making expression nodes know about model declarations. Keep expression nodes simple. The expression module should define the symbol type, but not `Param`, `Data`, or the decorator.

---

## Phase 4C: Implement declaration classes

File: `src/jaxstanv5/model/core.py`

Replace the old `Model` protocol with three declaration dataclasses.

Use a tiny generator in this file for declaration symbols:

```python
from itertools import count

_SYMBOL_IDS = count()


def next_symbol() -> UnresolvedSymbol:
    return UnresolvedSymbol(next(_SYMBOL_IDS))
```

```python
@dataclass(frozen=True)
class Param:
    distribution: Distribution
    constraint: Constraint | None = None
    size: Data | DataRef | int | None = None
    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)
```

```python
@dataclass(frozen=True)
class Data:
    symbol: UnresolvedSymbol = field(default_factory=next_symbol, init=False, repr=False)
```

```python
@dataclass(frozen=True)
class Observed:
    distribution: Distribution
```

Important:

- `Param` and `Data` are declarations.
- `ParamRef` and `DataRef` are expression nodes.
- Do not merge these concepts.

Each declaration should expose:

```python
def ref(self) -> ParamRef | DataRef:
    ...
```

Then delegate operators to that ref:

```python
def __add__(self, other: object) -> BinOp:
    return self.ref() + other
```

Implement for:

- `+`
- `-`
- `*`
- `/`
- reverse versions
- `[]`

This is the only reason declarations need operator methods.

---

## Phase 4D: Implement metadata types

File: `src/jaxstanv5/model/decorator.py`

Create:

```python
@dataclass(frozen=True)
class ModelMeta:
    params: dict[str, Param]
    data_slots: list[str]
    observed_name: str
    observed: Observed
    expressions: dict[str, ExprNode]
```

Rules:

- `data_slots` should include only explicit `Data()` declarations.
- `observed_name` is separate.
- `bind(...)` should require both `data_slots` and `observed_name`.

This keeps metadata easier to understand:

```text
x = Data()       -> data_slots = ["x"]
y = Observed()   -> observed_name = "y"
```

---

## Phase 4E: Implement the decorator in two passes

File: `src/jaxstanv5/model/decorator.py`

Use two passes over `cls.__dict__`.

### Pass 1: collect declarations and build symbol table

```python
params: dict[str, Param] = {}
data_slots: list[str] = []
symbols: dict[UnresolvedSymbol, str] = {}

for name, value in cls.__dict__.items():
    if isinstance(value, Param):
        params[name] = value
        symbols[value.symbol] = name
    elif isinstance(value, Data):
        data_slots.append(name)
        symbols[value.symbol] = name
```

### Pass 2: collect observed declaration and expressions

```python
observed_name: str | None = None
observed: Observed | None = None
expressions: dict[str, ExprNode] = {}

for name, value in cls.__dict__.items():
    if isinstance(value, Observed):
        observed_name = name
        observed = normalize_observed(value, symbols)
    elif is_expr_node(value):
        expressions[name] = normalize_expr(value, symbols)
```

Validation:

- exactly one `Observed`
- no missing observed variable
- no duplicate observed declaration

---

## Phase 4F: Normalize symbols to names

The decorator must convert `UnresolvedSymbol` values into real names.

Example before normalization:

```python
BinOp(
    "+",
    ParamRef(UnresolvedSymbol(0)),
    BinOp("*", ParamRef(UnresolvedSymbol(1)), DataRef(UnresolvedSymbol(2))),
)
```

After normalization:

```python
BinOp("+", ParamRef("alpha"), BinOp("*", ParamRef("beta"), DataRef("x")))
```

Implement:

```python
def normalize_expr(expr: ExprNode, symbols: dict[UnresolvedSymbol, str]) -> ExprNode:
    match expr:
        case ParamRef(name) if isinstance(name, UnresolvedSymbol):
            return ParamRef(symbols[name])
        case DataRef(name) if isinstance(name, UnresolvedSymbol):
            return DataRef(symbols[name])
        case ConstNode():
            return expr
        case BinOp(op, left, right):
            return BinOp(op, normalize_expr(left, symbols), normalize_expr(right, symbols))
        case IndexOp(base, index):
            return IndexOp(normalize_expr(base, symbols), normalize_expr(index, symbols))
```

If avoiding `match`, use explicit `isinstance` branches.

Also normalize distribution fields.

For `Normal(mu, sigma)`, `mu` may be a `BinOp` and `sigma` may be a `Param` declaration.

So implement:

```python
def normalize_distribution(dist: Distribution, symbols: dict[UnresolvedSymbol, str]) -> Distribution:
    # For dataclass distributions only.
    # For each field:
    # - Param -> ParamRef(name)
    # - Data -> DataRef(name)
    # - ExprNode -> normalize_expr(...)
    # - scalar -> unchanged
```

For this project, distributions are dataclasses, so rebuilding with `type(dist)(**normalized_fields)` is acceptable.

---

## Phase 4G: Normalize parameter declarations

Each collected `Param` must be normalized too, because hierarchical priors may contain references:

```python
alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
```

Before normalization:

```python
Normal(loc=Param(...), scale=Param(...))
size=Data(...)
```

After normalization:

```python
Normal(loc=ParamRef("alpha_pop"), scale=ParamRef("sigma_alpha"))
size=DataRef("n_groups")
```

Implement:

```python
def normalize_param(param: Param, symbols: dict[UnresolvedSymbol, str]) -> Param:
    return Param(
        distribution=normalize_distribution(param.distribution, symbols),
        constraint=param.constraint,
        size=normalize_size(param.size, symbols),
    )
```

`normalize_size` rules:

- `None` -> `None`
- `int` -> same int
- `Data` -> `DataRef(real_name)`
- `DataRef(UnresolvedSymbol(...))` -> `DataRef(real_name)`

---

## Phase 4H: Attach metadata and `bind`

After normalization:

```python
meta = ModelMeta(...)
cls._model_meta = meta
cls.bind = classmethod(make_bind(meta))
return cls
```

Static type checkers may not know decorated classes gain attributes. That is acceptable for runtime, but tests may need `getattr(...)` or a narrow cast later if `ty` complains.

If `ruff` complains about `setattr`, either:

```python
cls._model_meta = meta
cls.bind = classmethod(make_bind(meta))
```

or use:

```python
setattr(cls, "_model_meta", meta)  # noqa: B010
setattr(cls, "bind", classmethod(make_bind(meta)))  # noqa: B010
```

Prefer direct assignment if `ty` allows it; otherwise use `setattr` with local noqa.

---

## Phase 4I: Implement `BoundModel`

File: `src/jaxstanv5/model/bound.py`

```python
@dataclass(frozen=True)
class BoundModel:
    meta: ModelMeta
    data: dict[str, jax.Array]
    param_shapes: dict[str, tuple[int, ...]]
    n_params: int
```

This is the explicit state transition:

```text
Declared model class -> BoundModel
```

Do not put compiler or sampler logic here.

---

## Phase 4J: Implement `bind(...)`

`bind(...)` should:

1. Require all explicit data slots.
2. Require the observed data by `observed_name`.
3. Reject extra data.
4. Convert all data to JAX arrays.
5. Resolve parameter shapes.
6. Return `BoundModel`.

Expected data names:

```python
expected = set(meta.data_slots + [meta.observed_name])
```

Shape resolution:

```python
if param.size is None:
    shape = ()
elif isinstance(param.size, int):
    shape = (param.size,)
elif isinstance(param.size, DataRef):
    shape = (int(bound_data[param.size.name]),)
```

Parameter count:

```python
()      -> 1
(3,)    -> 3
(2, 4)  -> 8
```

For now, only scalar and one-dimensional data-dependent sizes are needed.

---

## Phase 4K: Public exports

File: `src/jaxstanv5/model/__init__.py`

Export:

```python
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import ModelMeta, model

__all__ = ["BoundModel", "Data", "ModelMeta", "Observed", "Param", "model"]
```

File: `src/jaxstanv5/__init__.py`

Export only top-level DSL names:

```python
from jaxstanv5.model import Data, Observed, Param, model

__all__ = ["Data", "Observed", "Param", "model"]
```

---

## Phase 4L: Validation commands

Run only the focused tests first:

```bash
uv run pytest tests/test_model_declarations.py -q
```

Then run the model slice:

```bash
uv run ruff format .
uv run ruff check .
uv run ty check src/jaxstanv5/model src/jaxstanv5/__init__.py
uv run pytest \
  tests/test_model_declarations.py \
  tests/test_model_expr.py \
  tests/test_constraints_positive.py \
  tests/test_distributions_normal.py \
  -q
```

Do not expect full-project `ty` yet while the vertical-slice integration test still references APIs from later phases.

---

## Design invariants

Keep these true:

- `Param` and `Data` are declarations.
- `ParamRef` and `DataRef` are symbolic references.
- Class-body expressions use `UnresolvedSymbol` values.
- The `@model` decorator normalizes `UnresolvedSymbol` values into real names.
- `Observed` is not a `Data` slot, but `bind(...)` still requires observed data.
- `BoundModel` contains data and shapes only; no inference logic.
- No BlackJAX in model code.
- No compiler logic in model code.
- Prefer immutable dataclasses.
- Do not use `Any`.
