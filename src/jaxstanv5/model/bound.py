"""Bound model state after concrete data is attached."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from jaxstanv5.model.decorator import ModelMeta


@dataclass(frozen=True)
class BoundModel:
    """Model metadata plus concrete data and resolved parameter shapes."""

    meta: ModelMeta
    data: dict[str, jax.Array]
    param_shapes: dict[str, tuple[int, ...]]
    n_params: int
