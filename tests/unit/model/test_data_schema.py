"""Unit tests for public data declaration schemas."""

from __future__ import annotations

import pytest

from jaxstanv5.model._data_schema import DataDimSymbol, DataRankSchema, DataShapeSchema
from jaxstanv5.model.core import Data


def test_data_helpers_create_scalar_vector_matrix_and_array_schemas() -> None:
    n = Data.scalar()

    assert n.schema == DataShapeSchema(())
    assert Data.vector().schema == DataRankSchema(1)
    assert Data.vector(n).schema == DataShapeSchema((DataDimSymbol(n.symbol),))
    assert Data.matrix().schema == DataRankSchema(2)
    assert Data.matrix(n, 3).schema == DataShapeSchema((DataDimSymbol(n.symbol), 3))
    assert Data.array(rank=3).schema == DataRankSchema(3)
    assert Data.array(shape=(n, 3, 2)).schema == DataShapeSchema((DataDimSymbol(n.symbol), 3, 2))


def test_bare_data_constructor_is_rejected() -> None:
    with pytest.raises(TypeError, match="Data.scalar"):
        Data()


def test_data_array_requires_shape_or_rank_but_not_both() -> None:
    n = Data.scalar()

    with pytest.raises(TypeError, match="shape or rank"):
        Data.array()
    with pytest.raises(TypeError, match="either shape or rank"):
        Data.array(shape=(n,), rank=1)


def test_data_shape_dimensions_must_be_non_negative_integers_or_scalar_data() -> None:
    x = Data.vector()

    with pytest.raises(TypeError, match="not bool"):
        Data.vector(True)
    with pytest.raises(ValueError, match="non-negative"):
        Data.vector(-1)
    with pytest.raises(TypeError, match="scalar data"):
        Data.vector(x)


def test_data_rank_must_be_non_negative_integer() -> None:
    with pytest.raises(TypeError, match="not bool"):
        Data.array(rank=True)
    with pytest.raises(ValueError, match="non-negative"):
        Data.array(rank=-1)


def test_data_matrix_requires_both_dimensions_or_neither() -> None:
    with pytest.raises(TypeError, match="both rows and cols"):
        Data.matrix(3)
    with pytest.raises(TypeError, match="both rows and cols"):
        Data.matrix(cols=3)
