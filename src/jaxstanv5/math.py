"""Symbolic math helpers for model declarations."""

from __future__ import annotations

from jaxstanv5.model._deferred import DeferredUnaryOp


def exp(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic exponential expression."""
    return DeferredUnaryOp("exp", value)


def sigmoid(value: object) -> DeferredUnaryOp:
    """Return a declaration-time symbolic logistic sigmoid expression."""
    return DeferredUnaryOp("sigmoid", value)
