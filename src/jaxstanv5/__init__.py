"""jaxstanv5 — the JAX/BlackJAX sampling backend for bayeswire models."""

from jaxstanv5.model import BoundModel, bind_model

__all__ = [
    "BoundModel",
    "bind_model",
]
