"""Unit tests for model declaration validation."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import pytest

from jaxstanv5 import model
from jaxstanv5._backends.jax.distributions import log_prob as distribution_log_prob
from jaxstanv5.constraints import Interval, Positive, UnitInterval
from jaxstanv5.constraints.core import Constraint
from jaxstanv5.distributions import (
    Beta,
    BetaBinomial,
    Binomial,
    Exponential,
    HalfNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    Truncated,
    Uniform,
)
from jaxstanv5.distributions.core import (
    Distribution,
    DistributionParameter,
    DistributionValue,
    LogProbability,
)
from jaxstanv5.model.core import Data, Observed, Param, PartiallyObserved
from jaxstanv5.model.decorator import _resolve_model_declaration


def param_size(value: object) -> Data | int | None:
    return cast(Data | int | None, value)


class OpaqueShiftedNormal:
    """Non-dataclass distribution used to test declaration validation."""

    def __init__(self, loc: DistributionParameter, scale: DistributionParameter) -> None:
        self.loc = loc
        self.scale = scale

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return distribution_log_prob(Normal(self.loc, self.scale), x)


class SlottedOpaqueShiftedNormal:
    """Slotted non-dataclass distribution used to test declaration validation."""

    __slots__ = ("loc", "scale")

    def __init__(self, loc: DistributionParameter, scale: DistributionParameter) -> None:
        self.loc = loc
        self.scale = scale

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return distribution_log_prob(Normal(self.loc, self.scale), x)


class PrivateSlottedOpaqueShiftedNormal:
    """Slotted distribution whose symbolic field uses Python name mangling."""

    __slots__ = ("__loc", "scale")

    def __init__(self, loc: DistributionParameter, scale: DistributionParameter) -> None:
        self.__loc = loc
        self.scale = scale

    def log_prob(self, x: DistributionValue) -> LogProbability:
        return distribution_log_prob(Normal(self.__loc, self.scale), x)


def test_bare_data_declaration_is_rejected_with_schema_guidance() -> None:
    with pytest.raises(TypeError, match="Data.scalar"):
        Data()


def test_private_model_declaration_resolution_accepts_prior_only_model() -> None:
    class PriorOnly:
        alpha = Param(Normal(0.0, 1.0))
        x = Data.vector()

    meta = _resolve_model_declaration(PriorOnly)

    assert tuple(meta.params) == ("alpha",)
    assert tuple(meta.data) == ("x",)
    assert meta.observed_nodes == ()


def test_model_declaration_rejects_array_like_expression_constants_with_guidance() -> None:
    offset = jnp.asarray([1.0, 2.0])

    with pytest.raises(TypeError) as exc_info:

        @model
        class ArrayConstantExpression:
            alpha = Param(Normal(0.0, 1.0), size=2)
            mu = alpha + offset
            y = Observed(Normal(mu, 1.0))

    message = str(exc_info.value)
    assert "Array-like constants are not supported in model declaration expressions" in message
    assert "Data.vector" in message
    assert "bind(...)" in message
    assert "ArrayImpl" not in message


def test_model_declaration_rejects_hidden_non_scalar_distribution_parameters() -> None:
    scale = jnp.asarray([1.0, 2.0])

    with pytest.raises(TypeError) as exc_info:

        @model
        class HiddenDistributionField:
            alpha = Param(Normal(0.0, 1.0), size=2)
            y = Observed(Normal(alpha, scale))

    message = str(exc_info.value)
    assert "Non-scalar distribution parameters" in message
    assert "Data.vector" in message
    assert "bind(...)" in message
    assert "ArrayImpl" not in message


def test_model_declaration_rejects_opaque_symbolic_distribution_parameters() -> None:
    with pytest.raises(TypeError) as exc_info:

        @model
        class OpaqueSymbolicDistributionField:
            mu = Param(Normal(0.0, 1.0))
            y = Observed(OpaqueShiftedNormal(mu, 1.0))

    message = str(exc_info.value)
    assert "Custom distributions with symbolic parameters must be dataclasses" in message
    assert "OpaqueShiftedNormal" in message
    assert "@dataclass" in message


def test_model_declaration_rejects_slotted_opaque_symbolic_distribution_parameters() -> None:
    with pytest.raises(TypeError) as exc_info:

        @model
        class SlottedOpaqueSymbolicDistributionField:
            mu = Param(Normal(0.0, 1.0))
            y = Observed(SlottedOpaqueShiftedNormal(mu, 1.0))

    message = str(exc_info.value)
    assert "Custom distributions with symbolic parameters must be dataclasses" in message
    assert "SlottedOpaqueShiftedNormal" in message


def test_model_declaration_rejects_private_slotted_opaque_symbolic_distribution_parameters() -> (
    None
):
    with pytest.raises(TypeError) as exc_info:

        @model
        class PrivateSlottedOpaqueSymbolicDistributionField:
            mu = Param(Normal(0.0, 1.0))
            y = Observed(PrivateSlottedOpaqueShiftedNormal(mu, 1.0))

    message = str(exc_info.value)
    assert "Custom distributions with symbolic parameters must be dataclasses" in message
    assert "PrivateSlottedOpaqueShiftedNormal" in message


def test_partially_observed_vector_rejects_rank_only_missing_index_schema() -> None:
    n = Data.scalar()
    n_obs = Data.scalar()
    observed_idx = Data.vector(n_obs)
    missing_idx = Data.vector()
    observed_values = Data.vector(n_obs)

    with pytest.raises(TypeError, match="missing_idx"):
        PartiallyObserved.vector(
            Normal(0.0, 1.0),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )


def test_partially_observed_vector_rejects_discrete_distribution() -> None:
    class DiscretePartial:
        n = Data.scalar()
        n_obs = Data.scalar()
        n_mis = Data.scalar()
        observed_idx = Data.vector(n_obs)
        missing_idx = Data.vector(n_mis)
        observed_values = Data.vector(n_obs)
        y = PartiallyObserved.vector(
            Poisson(1.0),
            length=n,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    with pytest.raises(TypeError, match="Discrete distributions cannot be partially observed"):
        _resolve_model_declaration(DiscretePartial)


def test_private_model_declaration_resolution_rejects_data_only_model() -> None:
    class DataOnly:
        x = Data.vector()

    with pytest.raises(ValueError, match="at least one stochastic declaration"):
        _resolve_model_declaration(DataOnly)


def test_private_model_declaration_resolution_accepts_multiple_observed_declarations() -> None:
    class MultipleObserved:
        y = Observed(Normal(0.0, 1.0))
        z = Observed(Normal(0.0, 1.0))

    meta = _resolve_model_declaration(MultipleObserved)

    assert tuple(node.name for node in meta.observed_nodes) == ("y", "z")


def test_model_resolution_requires_positive_constraint_for_positive_priors() -> None:
    class MissingConstraint:
        sigma = Param(Exponential(1.0))

    with pytest.raises(TypeError, match="support \\(0, inf\\).*Positive"):
        _resolve_model_declaration(MissingConstraint)


@pytest.mark.parametrize("distribution", [Exponential(1.0), HalfNormal(1.0)])
def test_private_model_declaration_resolution_accepts_positive_prior_constraint(
    distribution: Distribution,
) -> None:
    class ValidPositivePrior:
        sigma = Param(distribution, constraint=Positive())

    meta = _resolve_model_declaration(ValidPositivePrior)

    assert tuple(meta.params) == ("sigma",)


def test_private_model_declaration_resolution_rejects_implicit_truncated_normal_prior() -> None:
    class ImplicitTruncatedNormal:
        sigma = Param(Normal(0.0, 1.0), constraint=Positive())

    with pytest.raises(TypeError, match="Use Truncated"):
        _resolve_model_declaration(ImplicitTruncatedNormal)


def test_private_model_declaration_resolution_accepts_explicit_truncated_normal_prior() -> None:
    class ExplicitTruncatedNormal:
        sigma = Param(Truncated(Normal(0.0, 1.0), lower=0.0), constraint=Positive())

    meta = _resolve_model_declaration(ExplicitTruncatedNormal)

    assert tuple(meta.params) == ("sigma",)


def test_private_model_declaration_resolution_requires_matching_truncated_bounds() -> None:
    class MismatchedTruncatedNormal:
        sigma = Param(Truncated(Normal(0.0, 1.0), lower=-1.0), constraint=Positive())

    with pytest.raises(TypeError, match="effective support"):
        _resolve_model_declaration(MismatchedTruncatedNormal)


def test_private_model_declaration_resolution_rejects_truncated_prior_without_cdf_support() -> None:
    class TruncatedStudentT:
        sigma = Param(
            Truncated(StudentT(4.0, 0.0, 1.0), lower=0.0),
            constraint=Positive(),
        )

    with pytest.raises(TypeError, match="no supported CDF"):
        _resolve_model_declaration(TruncatedStudentT)


def test_private_model_declaration_resolution_rejects_truncated_observed_without_cdf_support() -> (
    None
):
    class TruncatedStudentTObserved:
        y = Observed(Truncated(StudentT(4.0, 0.0, 1.0), lower=0.0))

    with pytest.raises(TypeError, match="Observed.*no supported CDF"):
        _resolve_model_declaration(TruncatedStudentTObserved)


def test_private_model_declaration_resolution_accepts_truncated_observed_with_cdf_support() -> None:
    class TruncatedNormalObserved:
        y = Observed(Truncated(Normal(0.0, 1.0), lower=0.0))

    meta = _resolve_model_declaration(TruncatedNormalObserved)

    assert tuple(node.name for node in meta.observed_nodes) == ("y",)


def test_private_model_declaration_resolution_rejects_truncated_discrete_prior() -> None:
    class TruncatedDiscrete:
        count = Param(Truncated(Poisson(1.0), lower=0.0), constraint=Positive())

    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param priors"):
        _resolve_model_declaration(TruncatedDiscrete)


def test_private_model_declaration_resolution_rejects_truncated_discrete_partially_observed() -> (
    None
):
    class TruncatedDiscretePartial:
        observed_values = Data.vector(0)
        observed_idx = Data.vector(0)
        missing_idx = Data.vector(1)
        y = PartiallyObserved.vector(
            Truncated(Poisson(1.0), lower=0.0),
            length=1,
            observed=observed_values,
            observed_idx=observed_idx,
            missing_idx=missing_idx,
        )

    with pytest.raises(TypeError, match="Discrete distributions cannot be partially observed"):
        _resolve_model_declaration(TruncatedDiscretePartial)


def test_private_model_declaration_resolution_validates_truncated_effective_support() -> None:
    class MismatchedSupport:
        theta = Param(Truncated(Uniform(0.0, 1.0), lower=0.0), constraint=Positive())

    with pytest.raises(TypeError, match="effective support"):
        _resolve_model_declaration(MismatchedSupport)

    class MatchedSupport:
        theta = Param(Truncated(Uniform(0.0, 1.0), lower=0.0), constraint=UnitInterval())

    meta = _resolve_model_declaration(MatchedSupport)

    assert tuple(meta.params) == ("theta",)


def test_private_model_declaration_resolution_requires_unit_interval_for_beta_prior() -> None:
    class MissingConstraint:
        theta = Param(Beta(2.0, 2.0))

    with pytest.raises(TypeError, match="support \\(0, 1\\).*UnitInterval"):
        _resolve_model_declaration(MissingConstraint)


@pytest.mark.parametrize(
    ("constraint", "accepted"),
    [
        (UnitInterval(), True),
        (Interval(0.0, 1.0), True),
        (Positive(), False),
        (Interval(-1.0, 1.0), False),
    ],
)
def test_private_model_declaration_resolution_validates_uniform_prior_constraint(
    constraint: Constraint,
    accepted: bool,
) -> None:
    class UniformPrior:
        theta = Param(Uniform(0.0, 1.0), constraint=constraint)

    if accepted:
        meta = _resolve_model_declaration(UniformPrior)
        assert tuple(meta.params) == ("theta",)
    else:
        with pytest.raises(TypeError, match="Uniform prior has support"):
            _resolve_model_declaration(UniformPrior)


def test_private_model_declaration_resolution_matches_scalar_array_uniform_bounds() -> None:
    lower = jnp.asarray(0.25, dtype=jnp.float32)
    upper = jnp.asarray(0.75, dtype=jnp.float32)

    class ScalarArrayUniformPrior:
        theta = Param(Uniform(lower, upper), constraint=Interval(0.25, 0.75))

    meta = _resolve_model_declaration(ScalarArrayUniformPrior)

    assert tuple(meta.params) == ("theta",)


def test_private_model_declaration_resolution_compares_bounds_as_exact_python_floats() -> None:
    lower = jnp.asarray(0.1, dtype=jnp.float32)
    upper = jnp.asarray(0.9, dtype=jnp.float32)

    class Float32UniformPrior:
        theta = Param(Uniform(lower, upper), constraint=Interval(0.1, 0.9))

    with pytest.raises(TypeError, match="Uniform prior has support"):
        _resolve_model_declaration(Float32UniformPrior)


def test_private_model_declaration_resolution_rejects_near_but_unequal_uniform_bounds() -> None:
    class NearUniformPrior:
        theta = Param(Uniform(0.0, 1e-8), constraint=Interval(0.0, 1e-7))

    with pytest.raises(TypeError, match="Uniform prior has support"):
        _resolve_model_declaration(NearUniformPrior)


def test_private_model_declaration_resolution_rejects_shifted_large_uniform_bounds() -> None:
    class LargeUniformPrior:
        theta = Param(
            Uniform(1_000_000.0, 1_000_001.0),
            constraint=Interval(1_000_000.5, 1_000_001.5),
        )

    with pytest.raises(TypeError, match="Uniform prior has support"):
        _resolve_model_declaration(LargeUniformPrior)


def test_private_model_declaration_resolution_rejects_sub_ulp_uniform_bound_mismatch() -> None:
    class NearUnitUniformPrior:
        theta = Param(Uniform(0.0, 1.0 + 1e-12), constraint=UnitInterval())

    with pytest.raises(TypeError, match="Uniform prior has support"):
        _resolve_model_declaration(NearUnitUniformPrior)


def test_model_resolution_skips_symbolic_uniform_bound_constraint_check() -> None:
    class SymbolicUniformBounds:
        low = Data.scalar()
        high = Data.scalar()
        theta = Param(Uniform(low, high))

    meta = _resolve_model_declaration(SymbolicUniformBounds)

    assert tuple(meta.params) == ("theta",)


def test_private_model_declaration_resolution_rejects_discrete_parameter_priors() -> None:
    class DiscreteLatent:
        count = Param(Poisson(1.0))

    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param priors"):
        _resolve_model_declaration(DiscreteLatent)


def test_private_model_declaration_resolution_rejects_binomial_parameter_priors() -> None:
    class DiscreteLatent:
        count = Param(Binomial(10.0, 0.5))

    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param priors"):
        _resolve_model_declaration(DiscreteLatent)


def test_private_model_declaration_resolution_rejects_beta_binomial_parameter_priors() -> None:
    class DiscreteLatent:
        count = Param(BetaBinomial(10.0, 2.0, 3.0))

    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param priors"):
        _resolve_model_declaration(DiscreteLatent)


def test_private_model_declaration_resolution_rejects_negative_binomial_parameter_priors() -> None:
    class DiscreteLatent:
        count = Param(NegativeBinomial(1.0, 5.0))

    with pytest.raises(TypeError, match="Discrete distributions cannot be used as Param priors"):
        _resolve_model_declaration(DiscreteLatent)


def test_private_model_declaration_resolution_keeps_observed_name_out_of_data() -> None:
    class Fixture:
        x = Data.vector()
        y = Observed(Normal(0.0, 1.0))

    meta = _resolve_model_declaration(Fixture)

    assert tuple(meta.data) == ("x",)
    assert tuple(node.name for node in meta.observed_nodes) == ("y",)


def test_private_model_declaration_resolution_rejects_invalid_parameter_size_type() -> None:
    class InvalidSize:
        n = Data.scalar()
        alpha = Param(Normal(0.0, 1.0), size=param_size("large"))
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="declaration size"):
        _resolve_model_declaration(InvalidSize)


def test_private_model_declaration_resolution_rejects_bool_parameter_size() -> None:
    class BoolSize:
        alpha = Param(Normal(0.0, 1.0), size=True)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="Parameter size must be an integer, not bool"):
        _resolve_model_declaration(BoolSize)


def test_private_model_declaration_resolution_rejects_negative_literal_parameter_size() -> None:
    class NegativeSize:
        alpha = Param(Normal(0.0, 1.0), size=-1)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Parameter size must be non-negative"):
        _resolve_model_declaration(NegativeSize)


def test_private_model_declaration_resolution_rejects_aliased_param_declarations() -> None:
    class AliasedParam:
        alpha = Param(Normal(0.0, 1.0))
        beta = alpha
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        _resolve_model_declaration(AliasedParam)


def test_private_model_declaration_resolution_rejects_aliased_data_declarations() -> None:
    class AliasedData:
        x = Data.vector()
        z = x
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        _resolve_model_declaration(AliasedData)


def test_model_declaration_rejects_base_class_with_declarations() -> None:
    class Base:
        alpha = Param(Normal(0.0, 1.0))
        x = Data.vector()

    with pytest.raises(TypeError, match="must not use inheritance"):

        @model
        class Child(Base):
            y = Observed(Normal(0.0, 1.0))


def test_model_declaration_rejects_declaration_free_base_class() -> None:
    class Mixin:
        note = "no declarations here"

    with pytest.raises(TypeError, match="must not use inheritance"):

        @model
        class WithMixin(Mixin):
            y = Observed(Normal(0.0, 1.0))


def test_model_declaration_accepts_explicit_object_base() -> None:
    @model
    class ExplicitObjectBase(object):  # noqa: UP004
        y = Observed(Normal(0.0, 1.0))

    meta = _resolve_model_declaration(ExplicitObjectBase)

    assert tuple(node.name for node in meta.observed_nodes) == ("y",)


def test_private_model_declaration_resolution_rejects_aliased_observed_declarations() -> None:
    class AliasedObserved:
        y = Observed(Normal(0.0, 1.0))
        z = y

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        _resolve_model_declaration(AliasedObserved)
