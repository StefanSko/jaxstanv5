"""Bound model state after concrete data is attached."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bayeswire.model.decorator import ModelMeta
from bayeswire.model.dimensions import ResolvedModelDimensions


@dataclass(frozen=True)
class BoundModel:
    """Model metadata plus concrete backend data and resolved parameter shapes."""

    meta: ModelMeta
    data: Mapping[str, object]
    param_shapes: dict[str, tuple[int, ...]]
    n_params: int
    dimensions: ResolvedModelDimensions | None = None
