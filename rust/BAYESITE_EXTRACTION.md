# Bayesite extraction plan

This directory is intentionally prepared as the seed of a separate project named
**Bayesite**.

Bayesite should become a single-static-binary, agent-operable Bayesian workflow
engine for models encoded as serialized IR. `jaxstanv5` remains the current
reference Python producer; the Rust project should not be merged into
`jaxstanv5/main`.

Endgame command surface:

```sh
bayesite sample
bayesite diagnose
bayesite prior-predictive
bayesite recover
bayesite sbc
```

The intent is that an agent can download one binary and run the Bayesian
workflow from the CLI without Python, `uvx`, NumPy, or any runtime dependency
graph. Bayesite should be the SQLite-like artifact for Bayesian workflow, not a
library that requires a host language environment on the agent path.

## Source state

Use these refs as the historical anchor:

- jaxstanv5 `main`: `1ecff85` (`Merge PR #41 IR serialization`)
- Rust seed branch after merge/fixes: `561d4a9`
- This extraction-context commit copies docs/fixtures into `rust/` so a future
  split can start from this directory without losing project context.

Current IR v1 envelope remains:

```json
{"jaxstanv5_ir": 1, "model": {}}
```

Keep that for v1. Renaming the envelope to Bayesite is a versioned format
decision, not a repository-extraction cleanup.

## Context that must survive extraction

Required:

- `AGENTS.md` — Bayesite project instructions and invariants.
- `README.md` — current Rust runtime overview.
- `NOTICE` — provenance for ported numerical routines.
- `docs/invariants.md` — Bayesite runtime/workflow invariants.
- `docs/ir-format-v1.md` — current IR v1 wire format.
- `docs/ir-v1-tags.md` — built-in core-profile tag inventory.
- `docs/sampler.md` — NUTS/adaptation behavior notes.
- `tests/golden_ir/` — compatibility fixtures and hashes.
- `jaxstanv5-core/tests/data/` — Rust-owned numeric/diagnostic fixtures.
- `demo/` — first-class wasm/browser proof path.
- `justfile` — validation entry points.

Also preserve from the jaxstanv5 repository when useful:

- `scripts/generate_special_fn_table.py` — regenerates the Rust special-function
  reference table. Path constants must be adjusted after layout normalization.
- `scripts/check_rust_backend_posterior.py` — optional cross-backend posterior
  comparison. It assumes a colocated jaxstanv5 checkout and should become an
  optional conformance tool, not a core dependency.
- `tests/integration/test_rust_backend.py` — useful as a reference for the same
  cross-backend check, but not a Bayesite unit test as-is.

## Recommended first extracted layout

```text
README.md
AGENTS.md
NOTICE
Cargo.toml              # workspace
justfile
docs/
  invariants.md
  ir-format-v1.md
  ir-v1-tags.md
  sampler.md
crates/
  core/                 # moved from jaxstanv5-core
demo/
tests/
  golden_ir/
scripts/
  generate_special_fn_table.py
  check_rust_backend_posterior.py   # optional/conformance
```

Keep the crate two path components below the repo root (`crates/core`) or update
fixture paths in Rust tests. The current tests use paths equivalent to:

```text
{CARGO_MANIFEST_DIR}/../../tests/golden_ir/fixtures/*.json
```

## Mechanical extraction sketch

One possible history-preserving path:

```sh
git clone git@github.com:StefanSko/jaxstanv5.git bayesite-seed
cd bayesite-seed
git checkout claude/confident-heisenberg-js7ltl

# Keep the Rust seed and the copied context under rust/.
git filter-repo --force --path rust/

# Normalize layout.
git mv rust/README.md README.md
git mv rust/AGENTS.md AGENTS.md
git mv rust/NOTICE NOTICE
git mv rust/justfile justfile
git mv rust/docs docs
git mv rust/demo demo
git mv rust/tests tests
mkdir -p crates
git mv rust/jaxstanv5-core crates/core
rmdir rust
```

Then add a workspace root `Cargo.toml`:

```toml
[workspace]
members = ["crates/core"]
resolver = "3"
```

Update paths in `justfile`, `README.md`, demo glue, and scripts from
`jaxstanv5-core` to `crates/core`.

## Spec ownership policy

Do not create a separate spec repository immediately unless it has an owner and
release process. Until then:

- `jaxstanv5` is the canonical IR v1 producer and source of IR decisions.
- Bayesite vendors explicit snapshots of the IR docs and fixtures.
- Syncs should be visible commits: `Sync IR v1 fixtures from jaxstanv5 <commit>`.
- A future neutral spec project makes sense when IR v2 is designed, another
  producer appears, or compatibility matrices become release-critical.

## Bayesite direction

Bayesite should be framed as:

> a single-file, agent-operable, SQLite-like Bayesian workflow engine for
> serialized model IR.

Not as:

> a Rust backend hidden inside jaxstanv5.

Strong defaults:

- IR consumer/runtime first; no Rust model declaration language unless explicitly
  designed.
- Agent workflow commands are first-class: `sample`, `diagnose`,
  `prior-predictive`, `recover`, and `sbc`.
- NUTS only for posterior sampling.
- zero-dependency core.
- first-class wasm.
- stable CLI/wasm/protocol boundaries with machine-readable output.
- no hidden network, filesystem, clock, or entropy in core runtime.
- no Python, package manager, NumPy, or runtime dependency graph on the default
  agent execution path.

Keep a hard boundary between the runtime and workflow layers:

```text
core runtime:
  decode IR -> bind data -> logp/grad -> NUTS -> diagnostics

workflow CLI:
  sample -> diagnose -> prior-predictive -> recover -> sbc
```

The workflow CLI may own artifacts and command ergonomics. It must not pollute
core evaluation/sampling semantics.

## Validation after extraction

At minimum:

```sh
cargo fmt --check --manifest-path crates/core/Cargo.toml
cargo clippy --all-targets --manifest-path crates/core/Cargo.toml -- -D warnings
cargo test --manifest-path crates/core/Cargo.toml
cargo build --target wasm32-unknown-unknown --manifest-path crates/core/Cargo.toml
```

If `just` is kept:

```sh
just check
```

For optional conformance against the Python producer, run the adapted posterior
comparison script against a pinned `jaxstanv5` checkout.
