"""Core model types."""

from typing import Protocol


class Model(Protocol):
    """Protocol for a probabilistic model.

    A model describes the shape of a Bayesian inference problem:
    parameters, data variables, and the structure needed to bind
    data and run inference.

    This is intentionally a protocol — concrete model implementations
    may use different representations.
    """

    ...
