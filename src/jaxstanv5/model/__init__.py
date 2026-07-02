"""Model binding: the backend transition from bayeswire metadata to bound state."""

from jaxstanv5.model.binding import bind_model
from jaxstanv5.model.bound import BoundModel

__all__ = [
    "BoundModel",
    "bind_model",
]
