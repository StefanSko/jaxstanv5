"""Tests for explicit model declaration phase transitions."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Protocol, cast

from jaxstanv5.constraints.positive import Positive
from jaxstanv5.distributions.core import DistributionParameter
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model._pending import (
    PendingBinOp,
    PendingConst,
    PendingDataRef,
    PendingExprNode,
    PendingIndexOp,
    PendingParamRef,
    UnresolvedSymbol,
)
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import (
    ModelMeta,
    collect_pending_model,
    model,
    resolve_pending_model,
)
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ExprNode, IndexOp, ParamRef


class NormalLike(Protocol):
    loc: object
    scale: object


class DecoratedModelLike(Protocol):
    _model_meta: ModelMeta


class PhaseFixtureLike(Protocol):
    n_groups: Data
    group_idx: Data
    alpha_pop: Param
    sigma_alpha: Param
    alpha: Param


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


def pending_symbol(value: object) -> UnresolvedSymbol:
    assert isinstance(value, UnresolvedSymbol)
    return value


def make_phase_fixture() -> PhaseFixtureLike:
    class PhaseFixture:
        n_groups = Data()
        group_idx = Data()
        alpha_pop = Param(Normal(0.0, 1.0))
        sigma_alpha = Param(Normal(0.0, 1.0), constraint=Positive())
        alpha = Param(
            Normal(dist_param(alpha_pop), dist_param(sigma_alpha)),
            size=n_groups,
        )
        centered_alpha = alpha[group_idx] - 0.0
        mu = 1.0 + sigma_alpha * centered_alpha
        y = Observed(Normal(dist_param(mu), dist_param(sigma_alpha)))

    return cast(PhaseFixtureLike, PhaseFixture)


def normal_like(value: object) -> NormalLike:
    assert isinstance(value, Normal)
    return cast(NormalLike, value)


def assert_pending_tree(value: object) -> PendingExprNode:
    """Assert a value is a pure pending expression tree."""
    assert not isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp)
    assert isinstance(
        value,
        PendingParamRef | PendingDataRef | PendingConst | PendingBinOp | PendingIndexOp,
    )

    if isinstance(value, PendingParamRef | PendingDataRef):
        assert isinstance(value.name, UnresolvedSymbol)
    elif isinstance(value, PendingBinOp):
        assert_pending_tree(value.left)
        assert_pending_tree(value.right)
    elif isinstance(value, PendingIndexOp):
        assert_pending_tree(value.base)
        assert_pending_tree(value.index)

    return value


def assert_final_tree(value: object) -> ExprNode:
    """Assert a value is a pure final expression tree with named refs."""
    assert not isinstance(
        value,
        PendingParamRef | PendingDataRef | PendingConst | PendingBinOp | PendingIndexOp,
    )
    assert isinstance(value, ParamRef | DataRef | ConstNode | BinOp | IndexOp)

    if isinstance(value, ParamRef | DataRef):
        assert isinstance(value.name, str)
    elif isinstance(value, BinOp):
        assert_final_tree(value.left)
        assert_final_tree(value.right)
    elif isinstance(value, IndexOp):
        assert_final_tree(value.base)
        assert_final_tree(value.index)

    return value


def assert_no_raw_declarations(value: object) -> None:
    """Pending/final metadata must not embed raw class-body declarations."""
    assert not isinstance(value, Param | Data | Observed)

    if isinstance(value, PendingBinOp | BinOp):
        assert_no_raw_declarations(value.left)
        assert_no_raw_declarations(value.right)
    elif isinstance(value, PendingIndexOp | IndexOp):
        assert_no_raw_declarations(value.base)
        assert_no_raw_declarations(value.index)
    elif is_dataclass(value) and not isinstance(
        value,
        PendingParamRef | PendingDataRef | PendingConst | ParamRef | DataRef | ConstNode,
    ):
        for field in fields(value):
            assert_no_raw_declarations(getattr(value, field.name))


def test_collect_pending_model_normalizes_class_body_to_pending_phase() -> None:
    fixture = make_phase_fixture()
    pending_model = collect_pending_model(cast(type[object], fixture))

    assert list(pending_model.params) == [
        "alpha_pop",
        "sigma_alpha",
        "alpha",
    ]
    assert pending_model.data_slots == ["n_groups", "group_idx"]
    assert pending_model.observed_name == "y"
    assert set(pending_model.expressions) == {"centered_alpha", "mu"}

    assert pending_model.symbols[pending_symbol(fixture.n_groups.symbol)] == "n_groups"
    assert pending_model.symbols[pending_symbol(fixture.group_idx.symbol)] == "group_idx"
    assert pending_model.symbols[pending_symbol(fixture.alpha_pop.symbol)] == "alpha_pop"
    assert pending_model.symbols[pending_symbol(fixture.sigma_alpha.symbol)] == "sigma_alpha"
    assert pending_model.symbols[pending_symbol(fixture.alpha.symbol)] == "alpha"

    alpha = pending_model.params["alpha"]
    alpha_dist = normal_like(alpha.distribution)
    assert isinstance(alpha_dist.loc, PendingParamRef)
    assert alpha_dist.loc.name == pending_symbol(fixture.alpha_pop.symbol)
    assert isinstance(alpha_dist.scale, PendingParamRef)
    assert alpha_dist.scale.name == pending_symbol(fixture.sigma_alpha.symbol)
    assert isinstance(alpha.size, PendingDataRef)
    assert alpha.size.name == pending_symbol(fixture.n_groups.symbol)

    centered_alpha = assert_pending_tree(pending_model.expressions["centered_alpha"])
    assert isinstance(centered_alpha, PendingBinOp)
    assert centered_alpha.op == "-"
    assert isinstance(centered_alpha.left, PendingIndexOp)
    assert isinstance(centered_alpha.right, PendingConst)

    observed_dist = normal_like(pending_model.observed.distribution)
    assert_pending_tree(observed_dist.loc)
    assert isinstance(observed_dist.scale, PendingParamRef)

    assert_no_raw_declarations(pending_model)


def test_resolve_pending_model_produces_final_metadata_only() -> None:
    fixture = make_phase_fixture()
    pending_model = collect_pending_model(cast(type[object], fixture))
    meta = resolve_pending_model(pending_model)

    assert list(meta.params) == ["alpha_pop", "sigma_alpha", "alpha"]
    assert meta.data_slots == ["n_groups", "group_idx"]
    assert meta.observed_name == "y"
    assert set(meta.expressions) == {"centered_alpha", "mu"}

    alpha = meta.params["alpha"]
    alpha_dist = normal_like(alpha.distribution)
    assert alpha_dist.loc == ParamRef("alpha_pop")
    assert alpha_dist.scale == ParamRef("sigma_alpha")
    assert alpha.size == DataRef("n_groups")

    centered_alpha = assert_final_tree(meta.expressions["centered_alpha"])
    assert isinstance(centered_alpha, BinOp)
    assert centered_alpha.op == "-"
    assert isinstance(centered_alpha.left, IndexOp)
    assert centered_alpha.left.base == ParamRef("alpha")
    assert centered_alpha.left.index == DataRef("group_idx")
    assert centered_alpha.right == ConstNode(0.0)

    mu = assert_final_tree(meta.expressions["mu"])
    assert isinstance(mu, BinOp)
    assert mu.op == "+"
    assert mu.left == ConstNode(1.0)

    observed_dist = normal_like(meta.observed.distribution)
    assert_final_tree(observed_dist.loc)
    assert observed_dist.scale == ParamRef("sigma_alpha")

    assert_no_raw_declarations(meta)


def test_model_decorator_attaches_only_final_metadata() -> None:
    fixture = make_phase_fixture()
    decorated = cast(DecoratedModelLike, model(cast(type[object], fixture)))
    meta = decorated._model_meta

    assert_no_raw_declarations(meta)
    assert isinstance(meta.expressions["mu"], BinOp)
    assert_final_tree(meta.expressions["mu"])
