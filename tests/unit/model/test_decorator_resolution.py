"""Unit tests for model declaration resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import pytest

from jaxstanv5.distributions import Binomial, Normal, Poisson
from jaxstanv5.distributions.core import DistributionParameter
from jaxstanv5.math import exp, sigmoid
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import (
    _collect_declaration_symbols,
    _resolve_declaration_distribution,
    _resolve_declaration_expr,
    _resolve_declaration_size,
    _resolve_declarations,
)
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, IndexOp, ParamRef, UnaryOp


class NormalFields(Protocol):
    loc: object
    scale: object


class PoissonFields(Protocol):
    rate: object


class BinomialFields(Protocol):
    total_count: object
    probs: object


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


def normal_fields(value: object) -> NormalFields:
    assert isinstance(value, Normal)
    return cast(NormalFields, value)


def poisson_fields(value: object) -> PoissonFields:
    assert isinstance(value, Poisson)
    return cast(PoissonFields, value)


def binomial_fields(value: object) -> BinomialFields:
    assert isinstance(value, Binomial)
    return cast(BinomialFields, value)


@dataclass(frozen=True)
class UnknownExpr:
    value: object


def test_resolve_declarations_resolves_param_data_and_observed_inventory() -> None:
    n = Data()
    alpha = Param(Normal(0.0, 1.0), size=n)

    class Fixture:
        n_groups = n
        alpha_offset = alpha
        y = Observed(Normal(dist_param(alpha + n), 1.0))

    symbols = _collect_declaration_symbols(Fixture)
    resolved = _resolve_declarations(Fixture, symbols)

    assert resolved.data_slots == ["n_groups"]
    assert tuple(node.name for node in resolved.observed_nodes) == ("y",)
    assert set(resolved.params) == {"alpha_offset"}
    assert resolved.params["alpha_offset"].size == DataRef("n_groups")
    assert normal_fields(resolved.observed_nodes[0].distribution).loc == BinOp(
        "+",
        ParamRef("alpha_offset"),
        DataRef("n_groups"),
    )


def test_resolve_declarations_resolves_multiple_observed_nodes() -> None:
    theta = Param(Normal(0.0, 1.0))
    x = Data()

    class Fixture:
        theta_value = theta
        x_error = x
        x_obs = Observed(Normal(dist_param(theta), dist_param(x)))
        y_obs = Observed(Normal(dist_param(theta + 1.0), 2.0))

    symbols = _collect_declaration_symbols(Fixture)
    resolved = _resolve_declarations(Fixture, symbols)

    assert tuple(node.name for node in resolved.observed_nodes) == ("x_obs", "y_obs")
    x_dist = normal_fields(resolved.observed_nodes[0].distribution)
    y_dist = normal_fields(resolved.observed_nodes[1].distribution)
    assert x_dist.loc == ParamRef("theta_value")
    assert x_dist.scale == DataRef("x_error")
    assert y_dist.loc == BinOp("+", ParamRef("theta_value"), ConstNode(1.0))
    assert y_dist.scale == ConstNode(2.0)


def test_resolve_declaration_expr_builds_final_tree_recursively() -> None:
    alpha = Param(Normal(0.0, 1.0))
    group_idx = Data()
    expr = alpha[group_idx] + 1.5

    resolved = _resolve_declaration_expr(
        expr,
        {alpha.symbol: "alpha", group_idx.symbol: "group_idx"},
    )

    assert resolved == BinOp(
        "+",
        IndexOp(ParamRef("alpha"), DataRef("group_idx")),
        ConstNode(1.5),
    )


def test_resolve_declaration_expr_handles_unary_negation() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    expr = -(alpha + x)

    resolved = _resolve_declaration_expr(
        expr,
        {alpha.symbol: "alpha", x.symbol: "x"},
    )

    assert resolved == UnaryOp("neg", BinOp("+", ParamRef("alpha"), DataRef("x")))


def test_resolve_declaration_expr_rejects_unknown_declaration_symbols() -> None:
    alpha = Param(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Unknown declaration symbol"):
        _resolve_declaration_expr(alpha, {})


def test_resolve_declaration_expr_rejects_unknown_expression_values() -> None:
    with pytest.raises(TypeError, match="declaration expression"):
        _resolve_declaration_expr(UnknownExpr("bad"), {})


def test_resolve_declaration_size_resolves_data_refs_and_preserves_static_sizes() -> None:
    n_groups = Data()
    symbols = {n_groups.symbol: "n_groups"}

    assert _resolve_declaration_size(None, symbols) is None
    assert _resolve_declaration_size(3, symbols) == 3
    assert _resolve_declaration_size(n_groups, symbols) == DataRef("n_groups")


def test_resolve_declaration_size_rejects_unknown_declaration_symbols() -> None:
    n_groups = Data()

    with pytest.raises(ValueError, match="Unknown declaration symbol"):
        _resolve_declaration_size(n_groups, {})


def test_distribution_resolution_handles_symbolic_fields() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    resolved_dist = normal_fields(
        _resolve_declaration_distribution(
            Normal(dist_param(alpha + x), 2.0),
            {alpha.symbol: "alpha", x.symbol: "x"},
        )
    )

    assert resolved_dist.loc == BinOp("+", ParamRef("alpha"), DataRef("x"))
    assert resolved_dist.scale == ConstNode(2.0)


def test_distribution_resolution_handles_symbolic_unary_function_fields() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    resolved_dist = poisson_fields(
        _resolve_declaration_distribution(
            Poisson(dist_param(exp(alpha + x))),
            {alpha.symbol: "alpha", x.symbol: "x"},
        )
    )

    assert resolved_dist.rate == UnaryOp("exp", BinOp("+", ParamRef("alpha"), DataRef("x")))


def test_distribution_resolution_handles_symbolic_sigmoid_fields() -> None:
    alpha = Param(Normal(0.0, 1.0))
    x = Data()
    resolved_dist = binomial_fields(
        _resolve_declaration_distribution(
            Binomial(10.0, dist_param(sigmoid(alpha + x))),
            {alpha.symbol: "alpha", x.symbol: "x"},
        )
    )

    assert resolved_dist.probs == UnaryOp("sigmoid", BinOp("+", ParamRef("alpha"), DataRef("x")))


def test_distribution_resolution_rejects_final_expr_fields() -> None:
    with pytest.raises(TypeError, match="Final expression nodes"):
        _resolve_declaration_distribution(
            Normal(dist_param(ParamRef("alpha")), 1.0),
            {},
        )
