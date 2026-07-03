# bayeswire migration plan

Status: implemented (see the definition-of-done checklist at the end for
the record of what landed and the two deliberate deviations). This document
sequences the extraction of the authoring/IR frontend out of `jaxstanv5`
into a new repository, **`bayeswire`**, and the follow-through in
`bayescycle`, `bayesite`, and CI. Each phase is independently
shippable within its repo and ends with a validation gate. Because the wire
envelope is renamed with no dual-key window, the *toolchain as a whole* is
end-to-end broken between Phase 2 (bayeswire emits `bayeswire_ir`) and Phase 5
(bayesite accepts it) — acceptable greenfield, so run Phases 2–5 as one
sequenced burst rather than letting the window linger.

## Motivation

The value proposition of this toolchain is: **a Bayesian workflow an agent can
run end-to-end through files, deterministically, with an audit trail.** Every
workflow step is a command that reads files and writes files; every run is
seeded and replayable; run directories are append-only; provenance records
what was actually done. The customer is an agent (and the human auditing it),
and the product is trustworthy *process*, not a sampler.

Three conclusions from that proposition drive this migration:

**The contract is the product.** The system's real interface is not any one
tool — it is the model language and the artifact formats the tools exchange.
Today that language (`jaxstanv5_ir`) is philosophically homeless: named after,
versioned by, and produced by one component that is simultaneously the
authoring frontend and a sampling backend. The spec is documented in two
copies that have already diverged, each repo tests against fixtures it
synthesized itself, and every design dispute defaults to the owning
component's convenience. Extraction inverts the ownership: bayeswire *is* the
language — spec, corpus, and conformance runner versioned together — and every
other component is a producer or consumer that provably conforms. One
normative copy of the truth; drift becomes a red CI job instead of an
archaeology finding.

**The audit trail needs the model as data, not as code.** If a run directory
bottoms out at "a hash of arbitrary Python", the provenance chain requires
re-executing untrusted code to interpret. The IR is what makes the model an
inspectable document: `meta_from_dict` runs no user code, so every downstream
phase — diagnose, posterior-predictive, a re-fit months later — proceeds from
the artifact alone. This is why the IR survives even though the multi-backend
portability argument alone never justified it: it is an audit-trail primitive
first, an interchange format second.

**Minimal trust surface, honestly stated.** "Zero dependencies" is a
mitigation, not a security property, and it currently protects only the
sampling step while the authoring leg drags in the full JAX dependency tree.
After this migration the default agent path — author a model, serialize IR,
sample with Bayesite — executes Python in an environment containing exactly
one stdlib-only package plus one auditable Rust binary. That is the honest,
defensible version of the original supply-chain claim: not "no supply-chain
danger", but *every step of the default path is small enough to audit, pinned
enough to reproduce, and deterministic enough to replay*.

The clean end state, stated as properties rather than repositories: one
language with exactly one normative definition; every producer and consumer
conformance-tested against the same corpus rather than against private
fixtures; a default execution path whose entire trust surface is enumerable in
one sentence; backends that are interchangeable exactly where the conformance
suite proves they are, and honestly asymmetric everywhere else; and a workflow
harness that composes all of it through files an agent can read, replay, and
be held accountable to.

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

Three decisions fixed up front:

- **This is a greenfield migration.** There are no external users. No
  deprecation warnings, no re-export shims, no transitional releases, no
  wire-format back-compat. Moved code is deleted at the source in the same
  change that lands it at the destination; consumers update in lockstep PRs.
  The discipline below (spec changelog, conformance CI, pinned tags) exists to
  protect the *new* baseline going forward, not to preserve the old one.
- **The wire envelope is renamed to `{"bayeswire_ir": 1, "model": ...}`.**
  Bayesite's `AGENTS.md` says the envelope must not be renamed *casually* and
  that a neutral envelope is a deliberate format decision — this migration is
  that deliberate decision, recorded in the bayeswire spec changelog and in
  bayesite's `AGENTS.md` (Phase 5). The `jaxstanv5_ir` key, its spec docs, and
  all fixtures carrying it are deleted, not maintained alongside.
- **bayeswire is the spec home.** The IR format doc, tag registry doc, the
  dimension-sidecar format, and the golden fixture corpus live there. This
  resolves the current spec fork (the jaxstanv5 and bayesite copies of
  `ir-format-v1.md` have already diverged textually) by deleting both copies
  and making every downstream copy a generated, hash-checked vendored file.

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
tags fail explicitly. All of these survive the envelope rename untouched — the
rename changes two envelope strings, not the node encoding. The
seed-compatibility section currently pins a jaxstanv5 commit and the old
envelope key; Phase 5 rewrites it to pin a bayeswire release tag and the
`bayeswire_ir` envelope.

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
3. **Flip `bind` ownership by deletion.** `@model` currently attaches a
   `.bind` classmethod whose body lazily imports the JAX backend, and `ir.py`
   imports `_make_bind` — the frontend reaching into the backend. Delete the
   attached `.bind` and `_make_bind` outright: `bind_model(model_cls, values)`
   (already public, backend-owned) becomes the only binding path, and
   `bindable_from_meta` returns pure metadata classes. Update every test and
   doc that calls `.bind(...)` in the same change. This is an explicit state
   transition owned by the layer that performs it — the phase-boundary rule
   from both AGENTS.md files — with no transitional alias.
4. **Delete the `_jax_enable_x64_loaded` sniff** in `decorator.py`
   (declaration-time validation that peeks at `sys.modules["jax"]`).
   Declaration semantics must not depend on import order; in a frontend
   package with no JAX this check is unimplementable, so resolve it now:
   validate Uniform bounds with pure-Python float comparison only.
5. Golden fixture bytes are unchanged by this phase — `bind` is not part of
   `ModelMeta`. The envelope rename happens in Phase 2, not here.

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
3. **Write the spec fresh as `bayeswire_ir` v1, first content commit.** Merge
   bayesite's stricter envelope rules (envelope fields appear exactly once;
   fields outside the envelope keys are malformed) with jaxstanv5's
   producer-side caveats into one normative `spec/ir-format-v1.md` +
   `spec/ir-v1-tags.md` + `spec/dimension-sidecar-v1.md`, all written against
   the `{"bayeswire_ir": 1, "model": ...}` envelope. The old `jaxstanv5_ir`
   docs are inputs to this rewrite, then deleted at their sources; nothing
   documents the old key.
4. **Regenerate the corpus under the new envelope.** Port the reference models
   from `tests/golden_ir/`, re-serialize with the new envelope, and commit the
   result as `corpus/` — the new baseline. Sanity check once during the port:
   old and new fixture bytes must differ *only* in the envelope key
   (structural diff), proving the rename smuggled in no encoding change. The
   old fixtures are then deleted, not archived. Keep the committed logp/grad
   reference values, explicitly labeled as JAX-oracle outputs (bayesite's
   correctness contract already treats them this way).
5. Add the conformance runner skeleton (see CI section): `produce` (reference
   models -> canonical bytes -> compare against corpus) is implementable
   immediately inside bayeswire; `consume` hooks are wired in Phases 3 and 5.
6. Tag `bayeswire v0.1.0` once its own gate (`ruff`/`ty`/`pytest` + no-JAX
   walk + produce-conformance) is green.

Gate: bayeswire CI green; the one-time structural diff against the old
fixtures shows the envelope key as the only delta.

## Phase 3 — jaxstanv5 consumes bayeswire

1. Add `bayeswire` as a pinned dependency (exact version, not a range — the
   backend and the language must move in lockstep until the spec stabilizes).
2. **Delete the moved modules and rewrite every import in the same change.**
   No re-export shims, no `DeprecationWarning`s: `from jaxstanv5.model import
   Param` becomes `from bayeswire.model import Param` across src, tests,
   scripts, and docs in one commit. rust-style-python's rule against
   convenience wrappers duplicating the happy path applies with no external
   users to soften it for — a shim would be pure liability.
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
`[inproc]`; one end-to-end walkthrough regenerated without engine wrappers or
staging workarounds.

## Phase 5 — bayesite alignment

1. **Update the decoder envelope check** in `crates/core/src/ir.rs` from
   `jaxstanv5_ir` to `bayeswire_ir` (two strings: the version-key lookup and
   its error message). No other decoder change — tags, field lists, and
   entry-array semantics are untouched by the rename. Old-envelope documents
   fail with the standard unsupported-version error; there is no dual-key
   acceptance window, because nothing produces the old key anymore.
2. Replace bayesite's `docs/ir-format-v1.md` / `docs/ir-v1-tags.md` with
   vendored copies generated from the bayeswire spec at a pinned release tag,
   plus a CI check that the vendored bytes hash-match that tag (bayesite
   stays zero-dependency and offline-capable: files, not package management).
3. Re-vendor `tests/golden_ir/fixtures/` from the regenerated bayeswire
   corpus, same hash-check.
4. Rewrite the `AGENTS.md` seed-compatibility section: compatible IR source
   becomes `bayeswire <tag>`; current wire envelope becomes
   `{"bayeswire_ir": 1, "model": ...}`; and the envelope-rename caution is
   updated to record that the rename *was* the deliberate format decision the
   old wording reserved, with a pointer to the bayeswire spec changelog entry.
5. The optional oracle path (`check_validation_ladder.py --posterior
   --jaxstanv5-path ...`) keeps pointing at a jaxstanv5 checkout — the oracle
   is the backend, not the language — but the fixtures it compares against
   come from the shared corpus.

Gate: `python3 scripts/check_validation_ladder.py` green, including the new
vendored-bytes hash check; wasm build green (a wasm failure is a project
failure, per bayesite invariants).

## Phase 6 — Cleanup

Most deletion already happened inside Phases 2–5 (greenfield: nothing is
deprecated, everything is removed at the moment of replacement). What remains
is the sweep for stragglers:

1. Grep all four repos for `jaxstanv5_ir`, `_make_bind`, `.bind(`, and the old
   import paths; every hit is either deleted or a historical reference in a
   changelog. The old envelope key must not appear outside the bayeswire spec
   changelog entry that records its removal.
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
  the retired `"jaxstanv5_ir"` key anywhere) to enforce the Phase 6 audit

Version discipline: consumers pin bayeswire by exact tag; bumping the pin is a
PR whose diff *is* the compatibility review. The spec carries its own version
(`bayeswire_ir` starts and stays at 1 until a real format change); package
versions move freely underneath it as long as produce-conformance bytes are
stable. Any change to canonical
bytes, tags, or field lists requires a spec changelog entry, regenerated
corpus, and a coordinated consumer-pin bump — exactly the "golden-file diffs
plus an IR version decision" rule both repos already state.

## Rollback

Greenfield simplifies this to git: Phases 1 and 2 are additive and
independently revertible; Phases 3–5 are ordinary commits in repos with no
external users, so rollback is `git revert` of the lockstep PRs, coordinated
in the reverse order they landed. No compatibility window needs managing in
either direction. After the first post-migration bayeswire tag is consumed
everywhere, rollback stops being meaningful — fix forward.

## Definition of done

- [x] bayeswire bootstrapped; produce-conformance green; no-JAX walk green.
      *Resolved after merge:* `bayeswire v0.1.0` was tagged on `main` and
      all consumer pins re-pointed to it (jaxstanv5 and bayescycle
      `pyproject.toml`, bayesite `BAYESWIRE_TAG` via the vendor script,
      bayesite-viz's dev pin).
- [x] jaxstanv5 depends on pinned bayeswire; old import paths, `.bind`, and
      `_make_bind` deleted — no shims ever existed
- [x] bayescycle default path installs no JAX; no-JAX CI job added and the
      real-engine end-to-end test runs against a locally built release
      bayesite binary
- [x] bayesite vendors spec + fixtures from the pinned bayeswire commit with
      hash checks (`BAYESWIRE_TAG`, `bayeswire-vendor.json`, new validation
      ladder rung); seed-compat section updated; validation ladder green.
      *Deviation, recorded honestly:* unifying the corpus surfaced that the
      language's core profile includes `Truncated`, which bayesite does not
      evaluate. Implementing it in Rust is a feature outside this migration;
      the gap is pinned as an explicit decode-failure test
      (`fixtures_eval.rs`) and the two Truncated-bearing fixtures are
      excluded from evaluation-side conformance until it lands.
- [x] exactly one normative copy of the wire spec exists, in bayeswire
      (downstream copies are generated, hash-checked bytes)
- [x] golden artifact corpus committed (`corpus/artifacts/`, per_draw_v2,
      generated by a release bayesite binary); bayesite_idata tests consume
      it from the installed bayeswire pin
- [x] envelope is `bayeswire_ir: 1` everywhere; the retired `jaxstanv5_ir`
      key survives only in the bayeswire spec changelog entry recording its
      removal, this plan document, and the historical records that describe
      the rename (bayesite `AGENTS.md` seed-compat note and
      `BAYESITE_EXTRACTION.md`), plus bayesite's byte-identical vendored
      copy of the spec
