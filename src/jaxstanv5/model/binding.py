"""The explicit authoring-to-backend transition: bind concrete data to a model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from bayeswire.model import attached_model_dimensions, model_meta

if TYPE_CHECKING:
    from jaxstanv5.model.bound import BoundModel


def bind_model(model_cls: object, values: Mapping[str, object]) -> BoundModel:
    """Bind concrete data values to a bayeswire model class with the JAX backend."""
    if not isinstance(values, Mapping):
        raise TypeError("Model bind values must be a mapping")
    meta = model_meta(model_cls)
    dimensions = attached_model_dimensions(model_cls)
    from jaxstanv5._backends.jax.binding import bind_model_meta

    return bind_model_meta(meta, values, dimensions=dimensions)
