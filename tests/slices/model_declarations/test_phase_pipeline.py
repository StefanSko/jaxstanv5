"""Final metadata structure for real ``@model`` declarations.

This slice intentionally uses decorated model classes instead of inspecting the
private deferred syntax capture. Unit tests cover declaration resolution helpers
directly; this module checks that a realistic declaration reaches the expected
final metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import cast

from jaxstanv5 import Data, Observed, Param, model
from jaxstanv5.constraints import Positive
from jaxstanv5.distributions import DistributionParameter, Normal
from jaxstanv5.model import ModelMeta
from jaxstanv5.model.core import Data as DataDeclaration
from jaxstanv5.model.core import Observed as ObservedDeclaration
from jaxstanv5.model.core import Param as ParamDeclaration
from jaxstanv5.model.expr import BinOp, ConstNode, DataRef, ExprNode, IndexOp, ParamRef


@dataclass(frozen=True)
class NormalFields:
    loc: object
    scale: object


def dist_param(value: object) -> DistributionParameter:
    return cast(DistributionParameter, value)


def normal_fields(value: object) -> NormalFields:
    assert isinstance(value, Normal)
    return NormalFields(loc=value.loc, scale=value.scale)


def model_meta_of(value: object) -> ModelMeta:
    meta = object.__getattribute__(value, "_model_meta")
    assert isinstance(meta, ModelMeta)
    return meta


def assert_final_tree(value: object) -> ExprNode:
    """Assert a value is a pure final expression tree with named refs."""
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
    """Final metadata must not embed raw class-body declarations."""
    assert not isinstance(value, ParamDeclaration | DataDeclaration | ObservedDeclaration)

    if isinstance(value, BinOp):
        assert_no_raw_declarations(value.left)
        assert_no_raw_declarations(value.right)
    elif isinstance(value, IndexOp):
        assert_no_raw_declarations(value.base)
        assert_no_raw_declarations(value.index)
    elif is_dataclass(value) and not isinstance(value, ParamRef | DataRef | ConstNode):
        for field in fields(value):
            assert_no_raw_declarations(getattr(value, field.name))


@model
class HierarchicalPhaseModel:
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


def test_model_decorator_attaches_expected_final_metadata() -> None:
    meta = model_meta_of(HierarchicalPhaseModel)

    assert list(meta.params) == ["alpha_pop", "sigma_alpha", "alpha"]
    assert meta.data_slots == ["n_groups", "group_idx"]
    assert meta.observed_name == "y"
    assert set(meta.expressions) == {"centered_alpha", "mu"}


def test_hierarchical_parameter_distribution_and_size_use_named_refs() -> None:
    meta = model_meta_of(HierarchicalPhaseModel)
    alpha = meta.params["alpha"]
    alpha_dist = normal_fields(alpha.distribution)

    assert alpha_dist.loc == ParamRef("alpha_pop")
    assert alpha_dist.scale == ParamRef("sigma_alpha")
    assert alpha.size == DataRef("n_groups")


def test_expressions_and_observed_distribution_are_final_expression_trees() -> None:
    meta = model_meta_of(HierarchicalPhaseModel)

    centered_alpha = assert_final_tree(meta.expressions["centered_alpha"])
    assert centered_alpha == BinOp(
        "-",
        IndexOp(ParamRef("alpha"), DataRef("group_idx")),
        ConstNode(0.0),
    )

    mu = assert_final_tree(meta.expressions["mu"])
    assert mu == BinOp(
        "+",
        ConstNode(1.0),
        BinOp("*", ParamRef("sigma_alpha"), meta.expressions["centered_alpha"]),
    )

    observed_dist = normal_fields(meta.observed.distribution)
    assert_final_tree(observed_dist.loc)
    assert observed_dist.scale == ParamRef("sigma_alpha")


def test_final_metadata_does_not_embed_raw_class_body_declarations() -> None:
    meta = model_meta_of(HierarchicalPhaseModel)

    assert_no_raw_declarations(meta)
