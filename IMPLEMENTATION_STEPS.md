# Implementation Steps for the Vertical Slice

Goal: make `tests/integration/test_vertical_slice.py` pass using a narrow end-to-end path:

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

And hierarchical parameters via:

```python
alpha = Param(Normal(alpha_pop, sigma_alpha), size=n_groups)
beta = Param(Normal(beta_pop, sigma_beta), size=n_groups)
```

## 1. Implement distributions

Files:

- `src/jaxstanv5/distributions/core.py`
- `src/jaxstanv5/distributions/normal.py`
- `src/jaxstanv5/distributions/__init__.py`

Tasks:

- Define a `Distribution` protocol with `log_prob(x)`.
- Implement immutable `Normal(loc, scale)`.
- Export `Normal` publicly.

Tests:

- `Normal(0, 1).log_prob(...)` matches expected values.
- Broadcasting works for vector inputs.

## 2. Implement constraints

Files:

- `src/jaxstanv5/constraints/core.py`
- `src/jaxstanv5/constraints/positive.py`
- `src/jaxstanv5/constraints/__init__.py`

Tasks:

- Define a `Constraint` protocol.
- Implement immutable `Positive()` using:
  - constrained to unconstrained: `log(x)`
  - unconstrained to constrained: `exp(y)`
  - Jacobian adjustment: `y`
- Export `Positive` publicly.

Tests:

- Round trip `x -> log(x) -> exp(log(x))`.
- Jacobian term is correct.

## 3. Implement model expression nodes

File:

- `src/jaxstanv5/model/expr.py`

Tasks:

- Create symbolic expression nodes:
  - `ParamRef`
  - `DataRef`
  - `ConstNode`
  - `BinOp`
  - `IndexOp`
- Support arithmetic:
  - `+`
  - `-`
  - `*`
  - `/`
  - indexing via `[]`

Purpose:

- During class definition, expressions like `alpha + beta * x` build an expression tree instead of computing immediately.

## 4. Implement model declarations and `@model`

Files:

- `src/jaxstanv5/model/core.py`
- `src/jaxstanv5/model/decorator.py`
- `src/jaxstanv5/model/bound.py`
- `src/jaxstanv5/model/__init__.py`

Tasks:

- Implement declarations:
  - `Param(distribution, constraint=None, size=None)`
  - `Data()`
  - `Observed(distribution)`
- Implement `@model` decorator that collects:
  - parameters
  - data slots
  - observed variable name
  - intermediate expressions
- Attach `bind(...)` to decorated classes.
- Implement immutable `BoundModel` carrying:
  - model metadata
  - bound data
  - resolved parameter shapes
  - total unconstrained parameter count

Hierarchical behavior:

- `size=n_groups` stores a `DataRef`.
- At bind time, `n_groups` resolves to an integer.
- Group-level parameters get shape `(n_groups,)`.

## 5. Implement compiler

Files:

- `src/jaxstanv5/compiler/eval.py`
- `src/jaxstanv5/compiler/core.py`
- `src/jaxstanv5/compiler/__init__.py`

Tasks:

- Evaluate expression trees against concrete `params` and `data`.
- Compile a `BoundModel` to:
  - `log_density_fn(unconstrained_params)`
  - `initial_position(bound, seed)`
- In `log_density_fn`:
  1. Split flat unconstrained vector into named parameters.
  2. Apply inverse constraint transforms.
  3. Add Jacobian adjustments.
  4. Add prior log-probabilities.
  5. Evaluate observed likelihood.
  6. Return scalar total log-density.

Important design point:

- Distributions should remain simple.
- The compiler evaluates symbolic distribution arguments into concrete arrays before calling `log_prob`.

## 6. Implement NUTS sampling

Files:

- `src/jaxstanv5/inference/core.py`
- `src/jaxstanv5/inference/__init__.py`

Tasks:

- Implement immutable `SampleResult` with:
  - `samples: dict[str, jax.Array]`
  - `divergences: jax.Array`
- Implement `sample(bound, seed, num_warmup, num_samples)`.
- Use BlackJAX only internally:

```python
adapt = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
adapt_result, _ = adapt.run(key, initial_position, num_steps=num_warmup)
kernel = blackjax.nuts(log_prob_fn, **adapt_result.parameters)
```

- Run post-warmup sampling with `jax.lax.scan`.
- Unpack flat unconstrained samples into named parameter arrays.

## 7. Implement diagnostics

Files:

- `src/jaxstanv5/diagnostics/core.py`
- `src/jaxstanv5/diagnostics/__init__.py`

Tasks:

- Implement:
  - `rhat(samples) -> dict[str, jax.Array]`
  - `ess(samples) -> dict[str, jax.Array]`
- Keep diagnostics simple at first.
- Return one scalar diagnostic per parameter name.

## 8. Export public API

File:

- `src/jaxstanv5/__init__.py`

Export only:

```python
from jaxstanv5.model import Data, Observed, Param, model

__all__ = ["Data", "Observed", "Param", "model"]
```

Submodule exports:

```python
from jaxstanv5.distributions import Normal
from jaxstanv5.constraints import Positive
from jaxstanv5.inference import sample, SampleResult
from jaxstanv5.diagnostics import rhat, ess
```

## 9. Validate incrementally

After each phase:

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
```

Suggested order:

1. Unit tests for `Normal`.
2. Unit tests for `Positive`.
3. Unit tests for expression nodes.
4. Unit tests for `@model` collection.
5. Unit tests for `bind(...)` shape resolution.
6. Unit tests for compiled log density.
7. Unit tests for `sample(...)` on a tiny model.
8. Full integration test.

## 10. Main invariants

Keep these true throughout implementation:

- `@model` class declaration is the only happy path.
- `bind(...)` is the explicit transition from declared model to bound model.
- `sample(...)` runs NUTS only.
- BlackJAX remains internal.
- Hierarchical parameters are ordinary `Param`s with data-dependent size.
- Constraints are handled in unconstrained space with Jacobian adjustment.
- Use strict typing.
- Do not use `Any`.
- Prefer immutable dataclasses and explicit state transitions.
