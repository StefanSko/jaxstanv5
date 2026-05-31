"""Unit tests for model declaration validation."""

from __future__ import annotations

from typing import cast

import pytest

from jaxstanv5.distributions.beta_binomial import BetaBinomial
from jaxstanv5.distributions.binomial import Binomial
from jaxstanv5.distributions.normal import Normal
from jaxstanv5.distributions.poisson import Poisson
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import _resolve_model_declaration


def param_size(value: object) -> Data | int | None:
    return cast(Data | int | None, value)


def test_private_model_declaration_resolution_accepts_prior_only_model() -> None:
    class PriorOnly:
        alpha = Param(Normal(0.0, 1.0))
        x = Data()

    meta = _resolve_model_declaration(PriorOnly)

    assert tuple(meta.params) == ("alpha",)
    assert meta.data_slots == ["x"]
    assert meta.observed_nodes == ()


def test_private_model_declaration_resolution_rejects_data_only_model() -> None:
    class DataOnly:
        x = Data()

    with pytest.raises(ValueError, match="at least one stochastic declaration"):
        _resolve_model_declaration(DataOnly)


def test_private_model_declaration_resolution_accepts_multiple_observed_declarations() -> None:
    class MultipleObserved:
        y = Observed(Normal(0.0, 1.0))
        z = Observed(Normal(0.0, 1.0))

    meta = _resolve_model_declaration(MultipleObserved)

    assert tuple(node.name for node in meta.observed_nodes) == ("y", "z")


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


def test_private_model_declaration_resolution_keeps_observed_name_out_of_data_slots() -> None:
    class Fixture:
        x = Data()
        y = Observed(Normal(0.0, 1.0))

    meta = _resolve_model_declaration(Fixture)

    assert meta.data_slots == ["x"]
    assert tuple(node.name for node in meta.observed_nodes) == ("y",)


def test_private_model_declaration_resolution_rejects_invalid_parameter_size_type() -> None:
    class InvalidSize:
        n = Data()
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
        x = Data()
        z = x
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        _resolve_model_declaration(AliasedData)


def test_private_model_declaration_resolution_rejects_aliased_observed_declarations() -> None:
    class AliasedObserved:
        y = Observed(Normal(0.0, 1.0))
        z = y

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        _resolve_model_declaration(AliasedObserved)
