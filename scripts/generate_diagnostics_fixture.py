"""Generate the diagnostics reference fixture for the Rust core.

Run from the repository root:

    uv run scripts/generate_diagnostics_fixture.py

Deterministic AR(1) chains are generated with a fixed numpy seed and the
expected split R-hat / ESS values are computed with the same blackjax
functions the Python diagnostics use. The Rust tests
(`rust/jaxstanv5-core/tests/diagnostics_fixture.rs`) must reproduce these
numbers; the committed fixture is the contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from blackjax.diagnostics import (  # noqa: E402
    effective_sample_size,
    potential_scale_reduction,
)

OUT = Path(__file__).parent.parent / "rust" / "jaxstanv5-core" / "tests" / "data"


def ar1(rng: np.random.RandomState, n: int, rho: float, shift: float = 0.0) -> np.ndarray:
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = rho * x[i - 1] + rng.standard_normal()
    return x + shift


def case(name: str, chains: np.ndarray) -> dict[str, object]:
    arr = jnp.asarray(chains)
    out: dict[str, object] = {
        "name": name,
        "chains": [list(map(float, chain)) for chain in chains],
        "ess": float(effective_sample_size(arr)),
    }
    if chains.shape[0] > 1:
        out["rhat"] = float(potential_scale_reduction(arr))
    return out


def main() -> None:
    rng = np.random.RandomState(20240608)
    n = 250
    cases = [
        case("iid_like", np.stack([ar1(rng, n, 0.0) for _ in range(4)])),
        case("autocorrelated", np.stack([ar1(rng, n, 0.7) for _ in range(4)])),
        case(
            "shifted_means",
            np.stack([ar1(rng, n, 0.4, shift=float(c)) for c in range(3)]),
        ),
        case("single_chain", ar1(rng, n, 0.5)[None, :]),
        case("two_short_chains", np.stack([ar1(rng, 51, 0.3) for _ in range(2)])),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "diagnostics_fixture.json"
    path.write_text(json.dumps({"cases": cases}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
