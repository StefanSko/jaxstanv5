"""Unit tests for model declaration validation."""

from __future__ import annotations

from typing import cast

import pytest

from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import resolve_model_declaration


def param_size(value: object) -> Data | int | None:
    return cast(Data | int | None, value)


def test_resolve_model_declaration_rejects_missing_observed_declaration() -> None:
    class MissingObserved:
        alpha = Param(Normal(0.0, 1.0))
        x = Data()

    with pytest.raises(ValueError, match="exactly one Observed"):
        resolve_model_declaration(MissingObserved)


def test_resolve_model_declaration_rejects_multiple_observed_declarations() -> None:
    class MultipleObserved:
        y = Observed(Normal(0.0, 1.0))
        z = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="exactly one Observed"):
        resolve_model_declaration(MultipleObserved)


def test_resolve_model_declaration_keeps_observed_name_out_of_data_slots() -> None:
    class Fixture:
        x = Data()
        y = Observed(Normal(0.0, 1.0))

    meta = resolve_model_declaration(Fixture)

    assert meta.data_slots == ["x"]
    assert meta.observed_name == "y"


def test_resolve_model_declaration_rejects_invalid_parameter_size_type() -> None:
    class InvalidSize:
        n = Data()
        alpha = Param(Normal(0.0, 1.0), size=param_size("large"))
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="declaration size"):
        resolve_model_declaration(InvalidSize)


def test_resolve_model_declaration_rejects_bool_parameter_size() -> None:
    class BoolSize:
        alpha = Param(Normal(0.0, 1.0), size=True)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="Parameter size must be an integer, not bool"):
        resolve_model_declaration(BoolSize)


def test_resolve_model_declaration_rejects_negative_literal_parameter_size() -> None:
    class NegativeSize:
        alpha = Param(Normal(0.0, 1.0), size=-1)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Parameter size must be non-negative"):
        resolve_model_declaration(NegativeSize)


def test_resolve_model_declaration_rejects_aliased_param_declarations() -> None:
    class AliasedParam:
        alpha = Param(Normal(0.0, 1.0))
        beta = alpha
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        resolve_model_declaration(AliasedParam)


def test_resolve_model_declaration_rejects_aliased_data_declarations() -> None:
    class AliasedData:
        x = Data()
        z = x
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        resolve_model_declaration(AliasedData)
