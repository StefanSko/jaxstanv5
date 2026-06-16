"""Lazy accessors for the optional JAX backend dependency.

Authoring and IR modules import these proxies instead of importing JAX at module
load time. Accessing an attribute or calling a lazy function is a backend action
and imports the real JAX module then.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType


@dataclass(frozen=True)
class LazyModule:
    """Module proxy that imports the target only when an attribute is used."""

    module_name: str

    def __getattr__(self, name: str) -> object:
        module = import_module(self.module_name)
        return getattr(module, name)


@dataclass(frozen=True)
class LazyFunction:
    """Function proxy that imports the target module only when called."""

    module_name: str
    function_name: str

    def __call__(self, *args: object, **kwargs: object) -> object:
        module = import_module(self.module_name)
        function = getattr(module, self.function_name)
        return function(*args, **kwargs)


jax = LazyModule("jax")
jnp = LazyModule("jax.numpy")


def lazy_function(module_name: str, function_name: str) -> LazyFunction:
    """Return a lazy proxy for one backend function."""
    return LazyFunction(module_name, function_name)


def require_jax() -> ModuleType:
    """Import and return the optional JAX package for backend code."""
    return import_module("jax")


def require_jnp() -> ModuleType:
    """Import and return ``jax.numpy`` for backend code."""
    return import_module("jax.numpy")
