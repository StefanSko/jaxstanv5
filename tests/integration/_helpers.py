"""Shared helpers for integration tests."""

from jaxstanv5.model import bind_model as _bind_model
from jaxstanv5.model.bound import BoundModel


def bind_model(model_cls: object, **values: object) -> BoundModel:
    """Bind model data through the public explicit-transition API."""
    return _bind_model(model_cls, values)
