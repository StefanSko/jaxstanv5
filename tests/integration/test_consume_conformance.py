"""Consume-conformance against the bayeswire corpus.

The corpus is read from the installed bayeswire package, so this backend
proves it evaluates the *language's* fixtures, not a private copy. Each
fixture bundles an IR document, concrete bind data, and float64 JAX-oracle
log-density and gradient values; everything needed to run this test comes
from the artifact alone — no reference-model code is imported.

This test recomputes in the default test precision, so it compares with the
float32-level tolerance stated in the bayeswire spec (tolerances live in
``spec/ir-format-v1.md``, not here).
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import cast

import jax
import jax.numpy as jnp
import pytest
from bayeswire.ir import bindable_from_meta, meta_from_dict

from jaxstanv5.compiler import compile_log_density
from jaxstanv5.model import bind_model

CORPUS = files("bayeswire") / "corpus"

# Spec tolerance policy, float32 evaluation tier.
FLOAT32_RTOL = 1e-3
FLOAT32_ATOL = 1e-3

REGENERATE_HINT = (
    "Cross-backend corpus fixtures disagree with the compiled log density. "
    "If this change is deliberate, regenerate the fixtures against a "
    "bayeswire checkout with scripts/generate_ir_fixtures.py and review the "
    "corpus diff there."
)


def _fixture_names() -> list[str]:
    hashes = json.loads((CORPUS / "hashes.json").read_text(encoding="utf-8"))
    return sorted(hashes)


def _decode_bind_value(encoded: dict[str, object]) -> object:
    """Rebuild a bind value from its fixture encoding (dtype, shape, values)."""
    dtype = str(encoded["dtype"])
    shape = [int(dim) for dim in cast("list[int]", encoded["shape"])]
    flat: list[float | int] = list(cast("list[float]", encoded["values"]))
    if dtype.startswith("int"):
        flat = [int(value) for value in flat]
    return jnp.asarray(flat).reshape(shape)


def test_corpus_is_present_and_nonempty() -> None:
    names = _fixture_names()

    assert len(names) >= 6
    assert "linear_regression" in names


@pytest.mark.parametrize("name", _fixture_names())
def test_fixture_evaluations_match_compiled_log_density(name: str) -> None:
    fixture = json.loads((CORPUS / "fixtures" / f"{name}.json").read_text(encoding="utf-8"))

    assert fixture["name"] == name
    meta = meta_from_dict(fixture["ir"])
    bind_values = {
        data_name: _decode_bind_value(encoded) for data_name, encoded in fixture["data"].items()
    }
    bound = bind_model(bindable_from_meta(meta), bind_values)
    log_density = compile_log_density(bound)

    evaluations = fixture["evaluations"]
    assert len(evaluations) > 0
    for evaluation in evaluations:
        q = jnp.asarray(evaluation["q"], dtype=jnp.float32)
        assert q.shape == (bound.n_params,)
        expected_value = evaluation["log_density"]
        expected_gradient = jnp.asarray(evaluation["gradient"], dtype=jnp.float32)
        assert jnp.allclose(log_density(q), expected_value, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL), (
            REGENERATE_HINT
        )
        assert jnp.allclose(
            jax.grad(log_density)(q), expected_gradient, rtol=FLOAT32_RTOL, atol=FLOAT32_ATOL
        ), REGENERATE_HINT
