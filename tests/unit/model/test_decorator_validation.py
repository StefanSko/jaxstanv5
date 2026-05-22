"""Unit tests for model declaration validation."""

from __future__ import annotations

from typing import cast

import pytest

from jaxstanv5.distributions.normal import Normal
from jaxstanv5.model._pending import PendingDataRef
from jaxstanv5.model.core import Data, Observed, Param
from jaxstanv5.model.decorator import collect_pending_model


def param_size(value: object) -> Data | PendingDataRef | int | None:
    return cast(Data | PendingDataRef | int | None, value)


def test_collect_pending_model_rejects_missing_observed_declaration() -> None:
    class MissingObserved:
        alpha = Param(Normal(0.0, 1.0))
        x = Data()

    with pytest.raises(ValueError, match="exactly one Observed"):
        collect_pending_model(MissingObserved)


def test_collect_pending_model_rejects_multiple_observed_declarations() -> None:
    class MultipleObserved:
        y = Observed(Normal(0.0, 1.0))
        z = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="exactly one Observed"):
        collect_pending_model(MultipleObserved)


def test_collect_pending_model_keeps_observed_name_out_of_data_slots() -> None:
    class Fixture:
        x = Data()
        y = Observed(Normal(0.0, 1.0))

    pending = collect_pending_model(Fixture)

    assert pending.data_slots == ["x"]
    assert pending.observed_name == "y"


def test_collect_pending_model_rejects_invalid_parameter_size_type() -> None:
    class InvalidSize:
        n = Data()
        alpha = Param(Normal(0.0, 1.0), size=param_size("large"))
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="pending size"):
        collect_pending_model(InvalidSize)


def test_collect_pending_model_rejects_bool_parameter_size() -> None:
    class BoolSize:
        alpha = Param(Normal(0.0, 1.0), size=True)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(TypeError, match="Parameter size must be an integer, not bool"):
        collect_pending_model(BoolSize)


def test_collect_pending_model_rejects_negative_literal_parameter_size() -> None:
    class NegativeSize:
        alpha = Param(Normal(0.0, 1.0), size=-1)
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Parameter size must be non-negative"):
        collect_pending_model(NegativeSize)


def test_collect_pending_model_rejects_aliased_param_declarations() -> None:
    class AliasedParam:
        alpha = Param(Normal(0.0, 1.0))
        beta = alpha
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        collect_pending_model(AliasedParam)


def test_collect_pending_model_rejects_aliased_data_declarations() -> None:
    class AliasedData:
        x = Data()
        z = x
        y = Observed(Normal(0.0, 1.0))

    with pytest.raises(ValueError, match="Declaration aliases are not supported"):
        collect_pending_model(AliasedData)
