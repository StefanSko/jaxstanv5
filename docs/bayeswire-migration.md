# bayeswire migration plan

Status: proposal. This document sequences the extraction of the authoring/IR
frontend out of `jaxstanv5` into a new repository, **`bayeswire`**, and the
follow-through in `bayescycle`, `bayesite`, and CI. Each phase is independently
shippable and ends with a validation gate; no phase requires a cross-repo flag
day.

## End state

```text
bayeswire      Python authoring eDSL + IR + dimension sidecar + wire spec +
               golden fixture corpus. Stdlib only. Owns the language.

jaxstanv5      JAX/BlackJAX backend: bind -> compile log density -> NUTS ->
               diagnostics -> simulation -> InferenceData schema.
               Depends on bayeswire.

bayesite       Rust engine. Consumes the IR. Vendors spec + fixtures from
               bayeswire releases (byte-identical, CI-checked).

bayescycle     Workflow harness. Depends on bayeswire for authoring/IR always;
               depends on jaxstanv5 only via the optional [inproc] extra.

bayesite-viz   Unchanged surface. bayesite_idata gains a real golden artifact
               corpus sourced from bayeswire + bayesite releases.
```

Dependency direction is one-way: `jaxstanv5 -> bayeswire`, never the reverse.
`bayescycle -> bayeswire` (hard) and `bayescycle -> jaxstanv5` (optional).
`bayesite` has no package dependency on anything; it vendors files.

Two decisions fixed up front:

- **Package name is neutral (`bayeswire`); the wire envelope is not renamed.**
  The v1 envelope stays `{"jaxstanv5_ir": 1, "model": ...}`. Bayesite's
  `AGENTS.md` is explicit: *"Do not rename the v1 wire envelope casually. A
  neutral or Bayesite-branded envelope is a deliberate v2 format decision."*
  A `bayeswire_ir` envelope key is a v2 decision taken after this migration,
  not during it.
- **bayeswire is the spec home.** `docs/ir-format-v1.md`, `docs/ir-v1-tags.md`,
  the dimension-sidecar format, and `tests/golden_ir/` move there. This
  resolves the current spec fork (the jaxstanv5 and bayesite copies of
  `ir-format-v1.md` have already diverged textually) by making every other
  copy a generated, hash-checked vendored file.

## Governing principles

The migration must not weaken any repository's documented discipline. The
relevant sources, and what each one binds here:

**`.pi/skills/rust-style-python` (jaxstanv5, carried into bayeswire).**
Immutable or effectively immutable data; narrow, boring public APIs; explicit
state transitions; separate public contracts from backend machinery; typed
return values over ad hoc dicts; no hidden mutation, caching, or control flow;
loose input normalized quickly at public boundaries; no duck typing in core
semantics; validate before generalizing. Red flags that apply directly to this
migration: backend types in public APIs, registries without strong
justification, and abstractions that exist only for hypothetical futures.
bayeswire adopts this skill wholesale — it is *more* applicable to a wire-format
library than to a sampler.

**jaxstanv5 `AGENTS.md` / `docs/invariants.md`.** Strict typing, never `Any`;
phase boundaries as separate types and named transition functions; staged
red-green TDD from the current agreed state; test public behavior through
public APIs, no monkeypatching. The load-bearing invariant for this migration
already exists: importing `jaxstanv5.model`, `.distributions`, `.constraints`,
`.math`, or `.ir` must not import JAX or BlackJAX. The migration promotes that
invariant from a test to a package boundary.

**bayesite `AGENTS.md`.** The serialized `ModelMeta` is the backend boundary;
decoding executes no producer code; node tags and field lists are the wire
contract, not producer class names; entry-array order is semantic; consumers
hash received canonical bytes and never reserialize to hash; unknown non-core
tags fail explicitly. The seed-compatibility section pins a jaxstanv5 commit
and the envelope key — Phase 5 updates that section to pin a bayeswire release
tag instead.

**bayescycle `AGENTS.md` / `docs/invariants.md`.** bayescycle is a harness: it
must not invent model semantics, contain distribution math, or interpret
sampler telemetry. Public input is loose CLI input normalized quickly into
typed requests; the model.py -> IR transition is explicit and happens before
engine invocation. Phase 4 changes *which package* provides that transition,
not where it sits in the phase order.

**bayesite-viz `AGENTS.md`.** No knowledge of the bayesite/bayescycle/jaxstanv5
stack inside `bayesite_viz`; fit-format knowledge stays in `bayesite_idata`;
`bayesite_idata` remains extractable. This migration touches viz only through
Phase 7's golden artifact corpus, which `bayesite_idata` consumes as test
fixtures.

## Module inventory

Verified against the current import graph (exactly ten modules import JAX or
BlackJAX; everything else is stdlib-only).

**Moves to bayeswire:**

| Concern | Modules |
|---|---|
| Declaration eDSL | `model/core.py`, `model/expr.py`, `model/decorator.py` (minus `bind` attachment, see Phase 1), `model/dimensions.py`, `model/_data_schema.py`, `model/_deferred.py`, `model/_expression_errors.py` |
| Distribution metadata | `distributions/` (all modules — metadata-only dataclasses, including `Truncated`) |
| Constraint metadata | `constraints/` |
| Symbolic math namespace | `math.py` |
| IR codec + registry | `ir.py`, `_ir_registry.py` |
| Wire spec | `docs/ir-format-v1.md`, `docs/ir-v1-tags.md`, dimension-sidecar format |
| Golden corpus | `tests/golden_ir/` (fixtures move; parity *values* stay labeled as JAX-oracle reference outputs) |

**Stays in jaxstanv5:** `_backends/jax/` (binding, constraint transforms,
log-prob implementations), `compiler/`, `inference/`, `diagnostics/`,
`simulation/`, `interop/inferencedata.py`, `validation.py`, `data.py`
(`PartialVector` holds arrays), and `model/bound.py` — `BoundModel` is runtime
state holding backend arrays; it is not part of the authoring/IR boundary
(this matches the existing invariant wording).

## Phase 0 — Preconditions (complete)

- Public model hooks merged (#47): `model_meta(...)`, `is_model_class(...)`
  are JAX-free; `bind_model(...)` is the explicit JAX transition. Adapters no
  longer touch `_model_meta` or duck-type `bind`.
- Dimension sidecar round trip merged (#50 fix): `dimension_metadata_from_dict`
  plus `bindable_from_meta(meta, dimensions=...)`. The sidecar format now has
  both encode and decode paths, so it can be specified in bayeswire.
- `@model` rejects inheritance, and constrained-prior truncation is explicit
  via `Truncated(...)` — both reduce declaration-time ambiguity the frontend
  would otherwise have to carry across the split.

## Phase 1 — Harden the boundary in place (jaxstanv5)

Goal: make the split airtight while everything is still one repo, so the
extraction commit is a file move, not a redesign.

1. **Extend the import-boundary test** to assert the *entire* frontend module
   set (every module in the inventory above) imports with JAX and BlackJAX
   absent from `sys.modules`, not just the currently sampled subset.
2. **Publicize the private seams.** `compiler/core.py` and `inference/core.py`
   import `_resolved_free_values` / `_resolved_stochastic_sites` from
   `model/decorator.py`. Cross-repo private imports are how drift starts: add
   public accessors (`resolved_free_values(meta)`,
   `resolved_stochastic_sites(meta)`) to the frontend surface and migrate the
   backend callers. Per rust-style-python: typed return values, no incidental
   attribute access.
3. **Flip `bind` ownership.** `@model` currently attaches a `.bind`
   classmethod whose body lazily imports the JAX backend, and `ir.py` imports
   `_make_bind` — the frontend reaching into the backend. Replace with:
   `bind_model(model_cls, values)` (already public, backend-owned) is the
   supported path; the attached `.bind` delegates to it and is documented as
   deprecated; `bindable_from_meta` stops attaching any bind and returns pure
   metadata classes. This is an explicit state transition owned by the layer
   that performs it — the phase-boundary rule from both AGENTS.md files.
4. **Delete the `_jax_enable_x64_loaded` sniff** in `decorator.py`
   (declaration-time validation that peeks at `sys.modules["jax"]`).
   Declaration semantics must not depend on import order; in a frontend
   package with no JAX this check is unimplementable, so resolve it now:
   validate Uniform bounds with pure-Python float comparison only.
5. Regenerate golden fixtures if (and only if) step 3 changes serialized
   bytes; it should not — `bind` is not part of `ModelMeta`.

Gate: `ruff format --check`, `ruff check`, `ty check`, full pytest; golden
fixture bytes unchanged; the expanded import-boundary test green.

## Phase 2 — Bootstrap the bayeswire repository

1. Create `bayeswire` with the inventory above, package `bayeswire`, laid out
   as `bayeswire.model`, `bayeswire.distributions`, `bayeswire.constraints`,
   `bayeswire.math`, `bayeswire.ir`. Move history with `git filter-repo` if
   provenance matters; otherwise a clean import with a pointer commit.
2. Write bayeswire's own `AGENTS.md`, carrying over verbatim: the
   rust-style-python skill (vendored into the repo — it must not dangle as a
   cross-repo path reference), the strict-typing rules, the staged-TDD process,
   and the IR compatibility invariants currently duplicated between jaxstanv5
   and bayesite. Project identity: *"bayeswire is the model declaration
   language and wire format for the bayes* toolchain. It contains no
   inference, no distribution math, no plotting, no workflow orchestration,
   and imports nothing outside the standard library."* Invalid states the
   identity makes unrepresentable: a JAX import anywhere is a bug by
   definition, enforced by a repo-level test that walks every module.
3. **Reconcile the spec fork as the first content commit.** Merge bayesite's
   stricter envelope rules (envelope fields appear exactly once; fields
   outside `jaxstanv5_ir`/`model` are malformed) with jaxstanv5's
   producer-side caveats into one normative `spec/ir-format-v1.md` +
   `spec/ir-v1-tags.md` + `spec/dimension-sidecar-v1.md`. Record the merge as
   a spec changelog entry, since bayesite treats these rules as wire contract.
4. Move `tests/golden_ir/` fixtures in as `corpus/`. Keep the committed
   logp/grad reference values, explicitly labeled as JAX-oracle outputs
   (bayesite's correctness contract already treats them this way).
5. Add the conformance runner skeleton (see CI section): `produce` (reference
   models -> canonical bytes -> compare against corpus) is implementable
   immediately inside bayeswire; `consume` hooks are wired in Phases 3 and 5.
6. Tag `bayeswire v0.1.0` once its own gate (`ruff`/`ty`/`pytest` + no-JAX
   walk + produce-conformance) is green.

Gate: bayeswire CI green; canonical bytes of every corpus fixture identical to
the pre-move jaxstanv5 golden bytes.

## Phase 3 — jaxstanv5 consumes bayeswire

1. Add `bayeswire` as a pinned dependency (exact version, not a range — the
   backend and the language must move in lockstep until the spec stabilizes).
2. Delete the moved modules; replace the old import paths with **re-export
   shims** (`jaxstanv5.model` re-exports `bayeswire.model`, etc.), each
   emitting `DeprecationWarning`. One release of shims, no more — shims are a
   migration device, not an API (rust-style-python: no convenience wrappers
   that duplicate the happy path).
3. jaxstanv5's public story shrinks to its AGENTS.md identity: bind data,
   compile log densities, transforms/Jacobians, NUTS via BlackJAX, essential
   diagnostics, simulation, InferenceData schema. `bind_model` and
   `BoundModel` are its authoring-facing surface.
4. Move the logp/grad parity test to consume the bayeswire corpus as an
   installed package resource (or vendored copy with a hash check), so the
   backend proves it evaluates the *language's* fixtures, not its own copy.
5. Update `docs/invariants.md`: the authoring/IR boundary section now states
   that the boundary is the bayeswire package; the import-boundary test
   becomes "importing jaxstanv5's backend modules is the only thing that may
   import JAX".

Gate: full jaxstanv5 validation loop green against the pinned bayeswire;
the Stan/SBC smoke set for backend-boundary refactors (per AGENTS.md) run
once: `check_poisson_stan_posterior_reference`,
`check_beta_regression_stan_posterior_reference`,
`check_binomial_sbc_reference`, `check_ordinal_sbc_reference`.

## Phase 4 — bayescycle integration

1. Swap the authoring dependency: `bayescycle` depends on `bayeswire`
  (pinned by exact version — and fix the standing defect that
   `pyproject.toml` references a git URL with no rev). `jaxstanv5` moves
   entirely into the `[inproc]` extra.
2. The model loader uses bayeswire hooks (`is_model_class`, `model_meta`) for
   discovery and IR serialization; the in-process backend adapter uses
   jaxstanv5's `bind_model` + `sample`. This satisfies bayescycle's own module
   invariants: `_workflow.operations` loads models and writes workflow-owned
   inputs; `backends.jaxstanv5` remains the only module that touches the JAX
   runtime.
3. Consume the dims sidecar properly: `dims.json` is written from
   `dimension_metadata_to_dict` and — new capability — any run-directory
   consumer that reconstructs a model does so via
   `bindable_from_meta(meta, dimensions=dimension_metadata_from_dict(...))`.
4. Supply-chain payoff, stated in bayescycle's README: the default
   (`--backend bayesite`) path now executes `model.py` in an environment
   containing one stdlib-only package plus the Rust binary. No JAX on the
   default path, ever. This is the honest version of the original
   zero-dependency claim and should replace it in the docs.
5. bayescycle's boundary invariant is restated, not weakened: it still must
   not invent model semantics — it now cites bayeswire (not jaxstanv5) as the
   owner of authoring semantics, and jaxstanv5/bayesite as owners of sampler
   facts.

Gate: bayescycle validation loop green with `bayeswire` only (no JAX
installed) for the bayesite-backend test subset; full loop green with
`[inproc]`; one end-to-end walkthrough regenerated without shims or staging
workarounds.

## Phase 5 — bayesite alignment

1. Replace bayesite's `docs/ir-format-v1.md` / `docs/ir-v1-tags.md` with
   vendored copies generated from the bayeswire spec at a pinned release tag,
   plus a CI check that the vendored bytes hash-match that tag (bayesite
   stays zero-dependency and offline-capable: files, not package management).
2. Re-vendor `tests/golden_ir/fixtures/` from the bayeswire corpus, same
   hash-check. Decoder/evaluator behavior is unchanged — Phase 2 guaranteed
   byte-identical fixtures.
3. Update the `AGENTS.md` seed-compatibility section: compatible IR source
   becomes `bayeswire <tag>` instead of `jaxstanv5 main @ <commit>`. The
   envelope key statement stays exactly as written.
4. The optional oracle path (`check_validation_ladder.py --posterior
   --jaxstanv5-path ...`) keeps pointing at a jaxstanv5 checkout — the oracle
   is the backend, not the language — but the fixtures it compares against
   come from the shared corpus.

Gate: `python3 scripts/check_validation_ladder.py` green, including the new
vendored-bytes hash check; wasm build green (a wasm failure is a project
failure, per bayesite invariants).

## Phase 6 — Cleanup

1. Remove the jaxstanv5 re-export shims (one release after Phase 3); remove
   the deprecated `.bind` classmethod or reduce it to a one-line delegate with
   no independent behavior.
2. Delete jaxstanv5's copies of the spec docs; its `docs/` keeps only
   backend-owned documents (inferencedata compatibility, distribution
   coverage, invariants).
3. Sweep stale references: jaxstanv5 README authoring sections point to
   bayeswire; BACKLOG entries about decorator typing move to bayeswire;
   bayescycle walkthrough docs and `walkthrough-difficulties.md` items that
   this migration resolves are closed out explicitly.
4. Audit that no repo but bayeswire contains normative wire-format prose.
   Grep-level check in CI (see below), not a convention.

## Phase 7 — Golden artifact corpus (closes the loop with viz)

Not strictly part of the frontend extraction, but this is the moment the
corpus infrastructure exists, and it fixes the worst cross-repo test gap:
generate with a released bayesite binary a small set of real run artifacts
(`model.ir.json`, `data.json`, `dims.json`, `posterior.ndjson` with
`per_draw_v2`, `diagnostics.json`) from corpus models, and commit them to
bayeswire `corpus/artifacts/`. `bayesite_idata` replaces its synthetic inline
NDJSON dicts with these files; bayescycle's artifact-contract tests stop
skipping when a fake engine is all they have. bayesite-viz's AGENTS.md
boundary is respected: only `bayesite_idata` (the exporter) consumes them.

## CI alignment

Per-repo gates stay what they are (each repo's AGENTS.md validation loop; the
bayesite validation ladder). The migration adds one conformance axis, anchored
in bayeswire, plus hash checks in consumers:

**bayeswire CI (every push):**
- `ruff format --check`, `ruff check`, `ty check`, `pytest`
- no-JAX walk: import every module with `jax`/`blackjax` blocked from
  `sys.modules`; any import is a failure
- produce-conformance: declare every corpus reference model, serialize,
  byte-compare `canonical_bytes` output against `corpus/`
- spec-doc generation check: `ir-v1-tags.md` regenerated from the registry
  matches the committed file

**jaxstanv5 CI (every push):**
- standard validation loop against the *pinned* bayeswire version
- consume-conformance: decode every corpus fixture with `meta_from_dict`,
  evaluate logp/grad at the committed points, compare within the tolerance
  policy stated in the spec (tolerances live in the spec, not in this repo's
  tests)

**bayesite CI (every push):**
- validation ladder as today
- vendored-bytes check: `docs/ir-format-v1.md`, `docs/ir-v1-tags.md`,
  `tests/golden_ir/fixtures/` hash-match the pinned bayeswire tag recorded in
  a single `BAYESWIRE_TAG` file

**bayescycle CI (every push):**
- validation loop; plus a no-JAX job that installs only
  `bayescycle + bayeswire` and runs the bayesite-backend test subset with a
  release bayesite binary — this is the end-to-end test that does not exist
  today and would have caught the preflight and fingerprint breaks

**Cross-repo alignment job (in bayeswire, nightly + on tag):**
- checks out each consumer at its default branch, installs/vendors the
  bayeswire main HEAD, runs that consumer's conformance subset only
- purpose: early warning that a pending spec change breaks a consumer,
  *before* a tag is cut; a red nightly here blocks tagging, not merging
- also greps consumers for normative spec prose (`"wire contract"`,
  `"jaxstanv5_ir"` outside vendored files) to enforce the Phase 6 audit

Version discipline: consumers pin bayeswire by exact tag; bumping the pin is a
PR whose diff *is* the compatibility review. The spec carries its own version
(`jaxstanv5_ir` stays 1 throughout); package versions move freely underneath
it as long as produce-conformance bytes are stable. Any change to canonical
bytes, tags, or field lists requires a spec changelog entry, regenerated
corpus, and a coordinated consumer-pin bump — exactly the "golden-file diffs
plus an IR version decision" rule both repos already state.

## Rollback

Phases 1 and 2 are additive and independently revertible. Phase 3 is the
commitment point; its rollback is "repoint jaxstanv5 imports at the shims'
targets and restore the moved files from the bayeswire repo", which stays
cheap exactly as long as the Phase 1 boundary tests keep the code split-clean.
After Phase 5, rollback is a consumer-pin revert, never a code revert.

## Definition of done

- [ ] bayeswire tagged; produce-conformance green; no-JAX walk green
- [ ] jaxstanv5 depends on pinned bayeswire; shims removed after one release
- [ ] bayescycle default path installs no JAX; no-JAX CI job green with a
      release bayesite binary
- [ ] bayesite vendors spec + fixtures from a bayeswire tag with hash checks;
      seed-compat section updated; validation ladder green
- [ ] exactly one normative copy of the wire spec exists, in bayeswire
- [ ] golden artifact corpus committed; bayesite_idata tests consume it
- [ ] envelope key still `jaxstanv5_ir: 1`; any rename deferred to an explicit
      v2 spec decision
