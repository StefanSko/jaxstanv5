"""Tests keeping the cross-backend evaluation fixtures in sync with the compiler.

The fixtures bundle an IR document, concrete data, and float64 log-density
and gradient values for deterministic unconstrained points. Non-Python
backends differential-test against the recorded float64 values; this test
recomputes them in the default test precision, so it compares with a
float32-level tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from _ir_golden_models import GoldenIRCase, golden_ir_cases

from jaxstanv5.compiler import compile_log_density
from jaxstanv5.ir import bindable_from_meta, meta_from_dict
from jaxstanv5.model.bound import BoundModel

FIXTURE_DIR = Path(__file__).parent.parent / "golden_ir" / "fixtures"

REGENERATE_HINT = (
    "Cross-backend IR fixtures are out of date. If this change is deliberate, "
    "run scripts/generate_ir_fixtures.py and review the diff."
)


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


def _case_ids() -> list[str]:
    return [case.name for case in golden_ir_cases()]


@pytest.mark.parametrize("case", golden_ir_cases(), ids=_case_ids())
def test_fixture_matches_compiled_log_density(case: GoldenIRCase) -> None:
    fixture = json.loads((FIXTURE_DIR / f"{case.name}.json").read_text(encoding="utf-8"))

    assert fixture["name"] == case.name
    decoded = meta_from_dict(fixture["ir"])
    assert decoded == case.meta, REGENERATE_HINT

    rebuilt = cast(_BindableModel, bindable_from_meta(decoded))
    bound = rebuilt.bind(**case.bind_values)
    log_density = compile_log_density(bound)

    evaluations = fixture["evaluations"]
    assert len(evaluations) > 0
    for evaluation in evaluations:
        q = jnp.asarray(evaluation["q"], dtype=jnp.float32)
        assert q.shape == (bound.n_params,)
        expected_value = evaluation["log_density"]
        expected_gradient = jnp.asarray(evaluation["gradient"], dtype=jnp.float32)
        assert jnp.allclose(log_density(q), expected_value, rtol=1e-3, atol=1e-3), REGENERATE_HINT
        assert jnp.allclose(jax.grad(log_density)(q), expected_gradient, rtol=1e-3, atol=1e-3), (
            REGENERATE_HINT
        )
