"""Shared helpers for integration tests."""

from typing import Protocol, cast

from jaxstanv5.model.bound import BoundModel


class BindableModel(Protocol):
    """Model class after the runtime ``@model`` decorator attaches ``bind``."""

    def bind(self, **values: object) -> BoundModel:
        """Bind concrete model data."""
        ...


def bind_model(model_cls: object, **values: object) -> BoundModel:
    """Call runtime-attached ``bind`` through one explicit typed boundary."""
    return cast(BindableModel, model_cls).bind(**values)
