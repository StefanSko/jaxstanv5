"""Generate cross-backend IR evaluation fixtures in float64 (the JAX oracle).

Run from the repository root against a bayeswire checkout:

    uv run scripts/generate_ir_fixtures.py --bayeswire-path ../bayeswire

Each fixture bundles the IR document, concrete bind data, and log-density
plus gradient values at deterministic unconstrained points. The fixtures are
committed to the bayeswire corpus; non-Python backends parse the IR, bind
the data, and differential-test their evaluation against the recorded
float64 values within the tolerance policy stated in the bayeswire spec.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax

jax.config.update("jax_enable_x64", True)


def _q_points(n_params: int) -> list[list[float]]:
    """Return deterministic unconstrained evaluation points."""
    return [
        [0.0] * n_params,
        [0.1 * (i + 1) for i in range(n_params)],
        [0.5 * math.sin(1.0 + 0.7 * i) for i in range(n_params)],
    ]


def _encode_data(data: Mapping[str, object]) -> dict[str, object]:
    encoded: dict[str, object] = {}
    for name, raw_value in data.items():
        value = cast(jax.Array, raw_value)
        encoded[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": value.reshape(-1).tolist(),
        }
    return encoded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bayeswire-path",
        type=Path,
        required=True,
        help="Path to a bayeswire checkout; fixtures are written into its corpus",
    )
    args = parser.parse_args()

    bayeswire_root = args.bayeswire_path.resolve()
    fixture_dir = bayeswire_root / "src" / "bayeswire" / "corpus" / "fixtures"
    if not fixture_dir.is_dir():
        raise SystemExit(f"Not a bayeswire checkout (no corpus fixtures dir): {bayeswire_root}")
    sys.path.insert(0, str(bayeswire_root / "tests"))

    from bayeswire.ir import bindable_from_meta, meta_to_dict
    from conformance.reference_models import (  # ty: ignore[unresolved-import]
        reference_model_cases,
    )

    from jaxstanv5.compiler import compile_log_density
    from jaxstanv5.model import bind_model

    for case in reference_model_cases():
        rebuilt = bindable_from_meta(case.meta)
        bound = bind_model(rebuilt, dict(**case.bind_values))
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
        path = fixture_dir / f"{case.name}.json"
        path.write_text(
            json.dumps(fixture, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
