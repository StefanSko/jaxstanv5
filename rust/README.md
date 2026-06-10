# jaxstanv5-core — zero-dependency Rust sampling core

A sampling backend for [jaxstanv5](../README.md) IR documents in the spirit
of sqlite: one artifact, stdlib only, fully auditable, runnable in a sandbox
with no network and no filesystem beyond stdio — and, as a forcing function
for that discipline, runnable in a browser via WebAssembly.

`cargo tree` shows this crate and nothing else. That is the point: the JSON
parser, the PRNG, the special functions, the autodiff, the sampler, and the
wasm JS glue are all in-tree and reviewable.

## What it does

1. Parses the versioned IR v1 wire format
   ([docs/ir-format-v1.md](../docs/ir-format-v1.md)) — the **core profile**
   only. Tags outside [docs/ir-v1-tags.md](../docs/ir-v1-tags.md) fail with
   `UnknownNodeTag`, by design.
2. Binds concrete data and evaluates the log density and its gradient with
   its own reverse-mode AD over the closed IR op set.
3. Samples with multinomial NUTS (generalized U-turn criterion with
   across-subtree checks) and Stan-style warmup adaptation
   ([docs/sampler.md](docs/sampler.md)).
4. Emits draws and diagnostics in the **v0-provisional** NDJSON protocol via
   the `jstan` CLI or the wasm ABI.

## Correctness contract

- **Evaluation:** every fixture in `tests/golden_ir/fixtures/` must
  reproduce the committed JAX log density within rtol 1e-12 and the gradient
  within rtol 1e-10 (`tests/fixtures_eval.rs`).
- **Special functions:** pinned to a committed 400-digit mpmath table
  (`tests/data/special_fn_table.json`, generator
  `scripts/generate_special_fn_table.py`).
- **PRNG:** splitmix64 / xoshiro256++ pinned to Vigna's reference outputs.
- **Diagnostics:** split R-hat and ESS match `blackjax.diagnostics` value
  for value against a committed fixture
  (`tests/data/diagnostics_fixture.json`).
- **Sampler:** fixed-seed statistical tests against analytic targets
  (`tests/sampler_stats.rs`) plus a cross-backend posterior comparison over
  the whole golden corpus (`scripts/check_rust_backend_posterior.py`,
  `tests/integration/test_rust_backend.py`).

Draws are never bit-identical to BlackJAX (different RNGs); equivalence is
logp/grad parity at fixed points plus statistical agreement.

## Layout

| Path | Contents |
|---|---|
| `jaxstanv5-core/src/json.rs` | strict order-preserving JSON parser/writer |
| `jaxstanv5-core/src/ir.rs` | IR v1 core-profile decoder, typed errors |
| `jaxstanv5-core/src/tensor.rs` | f64 tensors, broadcasting, gather maps |
| `jaxstanv5-core/src/tape.rs` | reverse-mode AD over the closed op set |
| `jaxstanv5-core/src/density.rs` | distribution log densities (mirror the Python reference op for op) |
| `jaxstanv5-core/src/model.rs` | data binding, constraints, `Posterior::logp_grad` |
| `jaxstanv5-core/src/special.rs` | gammaln/digamma (Lanczos), erf/erfc/ndtr/ndtri (Cephes ports) |
| `jaxstanv5-core/src/linalg.rs` | Cholesky, triangular solves |
| `jaxstanv5-core/src/rng.rs` | splitmix64, xoshiro256++, polar normals |
| `jaxstanv5-core/src/nuts.rs`, `adapt.rs`, `sampler.rs` | NUTS, warmup adaptation, chain orchestration |
| `jaxstanv5-core/src/diagnostics.rs` | split R-hat, ESS |
| `jaxstanv5-core/src/protocol.rs` | v0-provisional NDJSON, wasm request handler |
| `jaxstanv5-core/src/wasm_abi.rs` | the only `unsafe` module: pointer/length shims |
| `jaxstanv5-core/src/bin/jstan.rs` | CLI (one thread per chain) |
| `demo/` | static browser demo (no bundler, no third-party code) |

The library is a **pure function**: no threads, no filesystem, no clock, no
OS entropy. Seeds are explicit arguments; parallelism belongs to callers
(CLI threads, web workers). This is what makes the wasm target free.

`#![deny(unsafe_code)]` is crate-wide; `wasm_abi.rs` is the single allowed
exception and only moves bytes.

## Building and testing

```sh
cd rust
just check        # fmt + clippy -D warnings + tests + wasm build
just wasm-release # optimized wasm artifact
```

Without `just`, run the four cargo commands in the `check` recipe directly.
The wasm target (`rustup target add wasm32-unknown-unknown`) is a build gate
from day one: the build breaking on wasm is a bug.

## The jstan CLI

```sh
cargo build --release --bin jstan
target/release/jstan sample \
    --model model_ir.json --data data.json \
    --seed 1 --chains 4 --warmup 1000 --draws 1000
```

stdout is NDJSON: a header (`"draws_format": "v0-provisional"`, parameter
shapes, packing order, settings), one object per draw with constrained
values keyed by parameter, and a trailer with per-chain divergences,
tree-depth histograms, step sizes, and cross-chain R-hat/ESS. Errors are a
single JSON object on stderr (`{"error": "<Kind>", "message": ...}`) with a
nonzero exit code; messages are written as repair instructions.

The `v0-provisional` marker is mandatory: the real fit-artifact format is
defined elsewhere, and nothing may grow load-bearing dependencies on this
one without noticing.

## Browser demo

```sh
cd rust
just wasm-release demo-assets
just demo   # serves the repo root on http://127.0.0.1:8000
```

Open <http://127.0.0.1:8000/rust/demo/>. The page loads a golden-corpus
model, runs one chain per web worker, and shows posterior summaries,
R-hat/ESS (computed by the same Rust core through the wasm ABI),
divergences, tree depths, and histograms. The only server involved is a
static file server.

## Provenance

Ported numerical routines (Cephes, Lanczos coefficients, xoshiro) are
documented in [NOTICE](NOTICE) and at each site.
