"""Generate cross-backend IR evaluation fixtures in float64.

Run from the repository root:

    uv run scripts/generate_ir_fixtures.py

Each fixture bundles the IR document, concrete bind data, and log-density
plus gradient values at deterministic unconstrained points. Non-Python
backends parse the IR, bind the data, and differential-test their
evaluation against the recorded float64 values.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Protocol, cast

import jax

jax.config.update("jax_enable_x64", True)

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from integration._ir_golden_models import golden_ir_cases  # noqa: E402
from jaxstanv5.compiler import compile_log_density  # noqa: E402
from jaxstanv5.ir import bindable_from_meta, meta_to_dict  # noqa: E402
from jaxstanv5.model.bound import BoundModel  # noqa: E402

FIXTURE_DIR = REPO_ROOT / "tests" / "golden_ir" / "fixtures"


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


def _q_points(n_params: int) -> list[list[float]]:
    """Return deterministic unconstrained evaluation points."""
    return [
        [0.0] * n_params,
        [0.1 * (i + 1) for i in range(n_params)],
        [0.5 * math.sin(1.0 + 0.7 * i) for i in range(n_params)],
    ]


def _encode_data(data: dict[str, jax.Array]) -> dict[str, object]:
    return {
        name: {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.reshape(-1).tolist(),
        }
        for name, value in data.items()
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for case in golden_ir_cases():
        rebuilt = cast(_BindableModel, bindable_from_meta(case.meta))
        bound = rebuilt.bind(**case.bind_values)
        log_density = compile_log_density(bound)
        gradient = jax.grad(log_density)

        evaluations: list[dict[str, object]] = []
        for q_values in _q_points(bound.n_params):
            q = jax.numpy.asarray(q_values, dtype=jax.numpy.float64)
            evaluations.append(
                {
                    "q": q_values,
                    "log_density": float(log_density(q)),
                    "gradient": gradient(q).tolist(),
                }
            )

        fixture = {
            "name": case.name,
            "ir": meta_to_dict(case.meta),
            "data": _encode_data(bound.data),
            "evaluations": evaluations,
        }
        path = FIXTURE_DIR / f"{case.name}.json"
        path.write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
