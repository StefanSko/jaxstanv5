"""Bound model state after concrete data is attached."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from jaxstanv5.model.decorator import ModelMeta

if TYPE_CHECKING:
    import jax

    type BoundDataValue = jax.Array
else:
    type BoundDataValue = object


@dataclass(frozen=True)
class BoundModel:
    """Model metadata plus concrete data and resolved parameter shapes."""

    meta: ModelMeta
    data: dict[str, BoundDataValue]
    param_shapes: dict[str, tuple[int, ...]]
    n_params: int
