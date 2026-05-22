"""Unit tests for pending-to-final model declaration resolution helpers."""

from __future__ import annotations

from typing import Protocol, cast

import pytest

from jaxstanv5.distributions.core import DistributionParameter
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model._pending import (
    PendingBinOp,
    PendingConst,
    PendingDataRef,
    PendingIndexOp,
    PendingParamRef,
    UnresolvedSymbol,
)
from jaxstanv5.model.core import Data, Param
from jaxstanv5.model.decorator import (
    normalize_distribution_to_pending,
    resolve_distribution,
    resolve_expr,
    resolve_size,
)
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef


class NormalFields(Protocol):
    loc: object
    scale: object


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


def normal_fields(value: object) -> NormalFields:
    assert isinstance(value, Normal)
    return cast(NormalFields, value)


def test_resolve_expr_resolves_each_pending_node_type_recursively() -> None:
    param_symbol = UnresolvedSymbol(1)
    data_symbol = UnresolvedSymbol(2)
    symbols = {param_symbol: "alpha", data_symbol: "group_idx"}

    pending = PendingBinOp(
        "+",
        PendingIndexOp(PendingParamRef(param_symbol), PendingDataRef(data_symbol)),
        PendingConst(1.5),
    )

    resolved = resolve_expr(pending, symbols)

    assert resolved == BinOp(
        "+",
        IndexOp(ParamRef("alpha"), DataRef("group_idx")),
        ConstNode(1.5),
    )


def test_resolve_expr_rejects_unknown_unresolved_symbols() -> None:
    with pytest.raises(ValueError, match="Unknown unresolved symbol"):
        resolve_expr(PendingParamRef(UnresolvedSymbol(999)), {})


def test_resolve_size_resolves_pending_data_refs_and_preserves_static_sizes() -> None:
    data_symbol = UnresolvedSymbol(1)
    symbols = {data_symbol: "n_groups"}

    assert resolve_size(None, symbols) is None
    assert resolve_size(3, symbols) == 3
    assert resolve_size(PendingDataRef(data_symbol), symbols) == DataRef("n_groups")


def test_resolve_size_rejects_unknown_unresolved_symbols() -> None:
    with pytest.raises(ValueError, match="Unknown unresolved symbol"):
        resolve_size(PendingDataRef(UnresolvedSymbol(999)), {})


def test_distribution_normalization_and_resolution_handle_symbolic_fields() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    pending_dist = normal_fields(
        normalize_distribution_to_pending(Normal(dist_param(alpha + x), 2.0))
    )

    assert isinstance(pending_dist.loc, PendingBinOp)
    assert isinstance(pending_dist.scale, PendingConst)

    resolved_dist = normal_fields(
        resolve_distribution(
            cast(Normal, pending_dist),
            {alpha.symbol: "alpha", x.symbol: "x"},
        )
    )

    assert resolved_dist.loc == BinOp("+", ParamRef("alpha"), DataRef("x"))
    assert resolved_dist.scale == ConstNode(2.0)
