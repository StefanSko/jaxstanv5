"""Import-boundary tests for JAX-free model authoring and IR export."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent


_BLOCK_JAX_IMPORTS = """
import importlib.abc
import sys

class BlockBackendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "jax" or fullname.startswith("jax."):
            raise ImportError(f"blocked backend import: {fullname}")
        if fullname == "blackjax" or fullname.startswith("blackjax."):
            raise ImportError(f"blocked backend import: {fullname}")
        return None

sys.meta_path.insert(0, BlockBackendImports())
"""


def _run_with_backend_imports_blocked(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_JAX_IMPORTS + textwrap.dedent(code)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_authoring_modules_import_without_jax_or_blackjax() -> None:
    result = _run_with_backend_imports_blocked(
        """
        import jaxstanv5
        import jaxstanv5.constraints
        import jaxstanv5.distributions
        import jaxstanv5.ir
        import jaxstanv5.math
        import jaxstanv5.model

        forbidden = [
            name for name in sys.modules
            if name == "jax" or name.startswith("jax.")
            or name == "blackjax" or name.startswith("blackjax.")
        ]
        assert forbidden == [], forbidden
        assert hasattr(jaxstanv5, "__all__")
        """
    )

    assert result.returncode == 0, result.stderr


def test_model_declaration_and_ir_serialization_run_without_jax_or_blackjax() -> None:
    result = _run_with_backend_imports_blocked(
        """
        from jaxstanv5 import Data, Observed, Param, model
        from jaxstanv5.distributions import Normal
        from jaxstanv5.ir import canonical_bytes, meta_to_dict

        @model
        class LinearAuthoringOnly:
            x = Data.vector()
            alpha = Param(Normal(0.0, 1.0))
            y = Observed(Normal(alpha, 1.0))

        document = meta_to_dict(LinearAuthoringOnly._model_meta)
        encoded = canonical_bytes(LinearAuthoringOnly._model_meta)

        assert document["jaxstanv5_ir"] == 1
        assert b"LinearAuthoringOnly" not in encoded
        forbidden = [
            name for name in sys.modules
            if name == "jax" or name.startswith("jax.")
            or name == "blackjax" or name.startswith("blackjax.")
        ]
        assert forbidden == [], forbidden
        """
    )

    assert result.returncode == 0, result.stderr


def test_jax_backend_dependencies_are_optional() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = tuple(pyproject["project"].get("dependencies", ()))
    forbidden_base = tuple(
        dependency
        for dependency in dependencies
        if dependency == "jax"
        or dependency.startswith("jax")
        or dependency == "blackjax"
        or dependency.startswith("blackjax")
    )

    assert forbidden_base == ()
    assert "jax" in pyproject["project"]["optional-dependencies"]
