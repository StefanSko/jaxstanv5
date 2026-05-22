"""Unit tests for model declaration resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import pytest

from jaxstanv5.distributions.core import DistributionParameter
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model.core import Data, Param
from jaxstanv5.model.decorator import (
    resolve_declaration_distribution,
    resolve_declaration_expr,
    resolve_declaration_size,
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


@dataclass(frozen=True)
class UnknownExpr:
    value: object


def test_resolve_declaration_expr_builds_final_tree_recursively() -> None:
    alpha = Param(Normal(0.0, 1.0))
    group_idx = Data()
    expr = alpha[group_idx] + 1.5

    resolved = resolve_declaration_expr(
        expr,
        {alpha.symbol: "alpha", group_idx.symbol: "group_idx"},
    )

    assert resolved == BinOp(
        "+",
        IndexOp(ParamRef("alpha"), DataRef("group_idx")),
        ConstNode(1.5),
    )


def test_resolve_declaration_expr_rejects_unknown_declaration_symbols() -> None:
    alpha = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Unknown declaration symbol"):
        resolve_declaration_expr(alpha, {})


def test_resolve_declaration_expr_rejects_unknown_expression_values() -> None:
    with pytest.raises(TypeError, match="declaration expression"):
        resolve_declaration_expr(UnknownExpr("bad"), {})


def test_resolve_declaration_size_resolves_data_refs_and_preserves_static_sizes() -> None:
    n_groups = Data()
    symbols = {n_groups.symbol: "n_groups"}

    assert resolve_declaration_size(None, symbols) is None
    assert resolve_declaration_size(3, symbols) == 3
    assert resolve_declaration_size(n_groups, symbols) == DataRef("n_groups")


def test_resolve_declaration_size_rejects_unknown_declaration_symbols() -> None:
    n_groups = Data()

    with pytest.raises(ValueError, match="Unknown declaration symbol"):
        resolve_declaration_size(n_groups, {})


def test_distribution_resolution_handles_symbolic_fields() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    resolved_dist = normal_fields(
        resolve_declaration_distribution(
            Normal(dist_param(alpha + x), 2.0),
            {alpha.symbol: "alpha", x.symbol: "x"},
        )
    )

    assert resolved_dist.loc == BinOp("+", ParamRef("alpha"), DataRef("x"))
    assert resolved_dist.scale == ConstNode(2.0)


def test_distribution_resolution_rejects_final_expr_fields() -> None:
    with pytest.raises(TypeError, match="Final expression nodes"):
        resolve_declaration_distribution(
            Normal(dist_param(ParamRef("alpha")), 1.0),
            {},
        )
