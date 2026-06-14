"""Differential test: the Rust subprocess backend against the JAX backend.

Builds ``jstan`` with cargo (skipped when no Rust toolchain is available),
samples a golden-corpus model over the subprocess protocol, and compares
posterior summaries against BlackJAX NUTS on the same model. Draws are
never bit-identical (different RNGs); the equivalence relation is
statistical agreement plus the logp/grad parity pinned by the fixture
conformance tests on the Rust side.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, cast

import jax
import pytest

from jaxstanv5.inference import sample
from jaxstanv5.ir import bindable_from_meta, meta_from_dict
from jaxstanv5.model.bound import BoundModel

REPO_ROOT = Path(__file__).parent.parent.parent
FIXTURE = REPO_ROOT / "tests" / "golden_ir" / "fixtures" / "linear_regression.json"
CRATE_DIR = REPO_ROOT / "rust" / "jaxstanv5-core"

NUM_WARMUP = 500
NUM_DRAWS = 1000
NUM_CHAINS = 2


class _BindableModel(Protocol):
    def bind(self, **values: object) -> BoundModel: ...


@pytest.fixture
def jax_x64() -> Iterator[None]:
    """Enable x64 only for this test and restore the global JAX setting."""
    previous = bool(jax.config.read("jax_enable_x64"))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture(scope="module")
def jstan_binary() -> Path:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available; cannot build the Rust backend")
    build = subprocess.run(
        ["cargo", "build", "--release", "--bin", "jstan"],
        cwd=CRATE_DIR,
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        pytest.fail(f"cargo build failed:\n{build.stderr}")
    return CRATE_DIR / "target" / "release" / "jstan"


def _run_jstan(binary: Path, tmp_path: Path) -> tuple[dict[str, object], list[dict], dict]:
    fixture = json.loads(FIXTURE.read_text())
    model_path = tmp_path / "model.json"
    data_path = tmp_path / "data.json"
    model_path.write_text(json.dumps(fixture["ir"]))
    data_path.write_text(json.dumps(fixture["data"]))

    result = subprocess.run(
        [
            str(binary),
            "sample",
            "--model",
            str(model_path),
            "--data",
            str(data_path),
            "--seed",
            "20240608",
            "--chains",
            str(NUM_CHAINS),
            "--warmup",
            str(NUM_WARMUP),
            "--draws",
            str(NUM_DRAWS),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"jstan failed: {result.stderr}"
    lines = result.stdout.splitlines()
    header = json.loads(lines[0])
    draws = [json.loads(line) for line in lines[1:-1]]
    trailer = json.loads(lines[-1])["trailer"]
    return header, draws, trailer


def _python_reference() -> dict[str, tuple[float, float]]:
    fixture = json.loads(FIXTURE.read_text())
    meta = meta_from_dict(fixture["ir"])
    rebuilt = cast(_BindableModel, bindable_from_meta(meta))
    bind_values = {
        name: spec["values"] if spec["shape"] else spec["values"][0]
        for name, spec in fixture["data"].items()
    }
    bound = rebuilt.bind(**bind_values)
    result = sample(
        bound,
        seed=11,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_DRAWS,
        num_chains=NUM_CHAINS,
    )
    summaries: dict[str, tuple[float, float]] = {}
    for name, values in result.samples.items():
        flat = values.reshape(-1)
        summaries[name] = (float(flat.mean()), float(flat.std()))
    return summaries


def test_rust_backend_matches_jax_posterior(
    jstan_binary: Path,
    tmp_path: Path,
    jax_x64: None,
) -> None:
    header, draws, trailer = _run_jstan(jstan_binary, tmp_path)

    assert header["draws_format"] == "v0-provisional"
    assert header["packing"] == ["alpha", "beta", "sigma"]
    assert len(draws) == NUM_CHAINS * NUM_DRAWS

    # Convergence of the Rust run itself.
    for name, rhat in trailer["rhat"].items():
        assert rhat < 1.05, f"{name}: rhat {rhat}"
    for name, ess in trailer["ess"].items():
        assert ess > 100, f"{name}: ess {ess}"
    for chain_stats in trailer["chains"]:
        assert chain_stats["divergences"] <= NUM_DRAWS * 0.01

    rust: dict[str, tuple[float, float]] = {}
    for name in ("alpha", "beta", "sigma"):
        values = [d["values"][name] for d in draws]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        rust[name] = (mean, math.sqrt(var))

    reference = _python_reference()
    for name, (ref_mean, ref_sd) in reference.items():
        rust_mean, rust_sd = rust[name]
        # Both runs have ESS in the hundreds, so each mean carries an MC
        # error of roughly sd/sqrt(ESS) ~ 0.06*sd; 0.25*sd catches real
        # disagreement with a wide false-positive margin.
        tolerance = 0.25 * ref_sd
        assert abs(rust_mean - ref_mean) < tolerance, (
            f"{name}: rust mean {rust_mean:.4f} vs jax mean {ref_mean:.4f} "
            f"(tolerance {tolerance:.4f})"
        )
        assert 0.7 < rust_sd / ref_sd < 1.4, f"{name}: rust sd {rust_sd:.4f} vs jax sd {ref_sd:.4f}"
