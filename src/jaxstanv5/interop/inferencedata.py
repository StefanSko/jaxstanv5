"""InferenceData-compatible schema adapter.

This module does not construct ArviZ, xarray, netCDF, or zarr objects.  It
returns typed groups with InferenceData-compatible group names, variable names,
dimensions, coordinates, and JAX arrays so downstream exporter packages can own
those dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import jax
import jax.numpy as jnp
from bayeswire.model.dimensions import CoordValue, ResolvedModelDimensions

from jaxstanv5.inference import SamplerResult
from jaxstanv5.model.bound import BoundModel

_CHAIN_DIM = "chain"
_DRAW_DIM = "draw"


@dataclass(frozen=True)
class InferenceDataVariable:
    """One named array plus its InferenceData dimension names."""

    dims: tuple[str, ...]
    values: jax.Array


@dataclass(frozen=True)
class InferenceDataGroup:
    """One InferenceData group represented as named variables."""

    variables: Mapping[str, InferenceDataVariable]


@dataclass(frozen=True)
class InferenceDataGroups:
    """InferenceData-compatible groups derived from a bound model and samples."""

    posterior: InferenceDataGroup
    sample_stats: InferenceDataGroup
    observed_data: InferenceDataGroup
    constant_data: InferenceDataGroup
    coords: Mapping[str, tuple[CoordValue, ...]]


def inferencedata_groups(bound: BoundModel, result: SamplerResult) -> InferenceDataGroups:
    """Build typed InferenceData-compatible groups without importing ArviZ.

    Posterior sample arrays keep jaxstanv5's constrained ``(chain, draw, ...)``
    layout.  NUTS diagnostics are mapped to conventional ArviZ ``sample_stats``
    names where the semantics are direct.
    """
    num_chains, num_draws = _chain_draw_shape(
        result.diagnostics.sampling.is_divergent,
        label="sample_stats.diverging",
    )
    _validate_posterior_sample_keys(bound, result)
    _validate_declared_value_dimension_names(bound.dimensions)
    coords: dict[str, tuple[CoordValue, ...]] = {
        _CHAIN_DIM: tuple(range(num_chains)),
        _DRAW_DIM: tuple(range(num_draws)),
    }
    _add_declared_coords(coords, bound.dimensions)
    dim_sizes: dict[str, int] = {_CHAIN_DIM: num_chains, _DRAW_DIM: num_draws}
    reserved_dims = _reserved_dimension_names(bound.dimensions)

    return InferenceDataGroups(
        posterior=_posterior_group(
            bound,
            result,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
            num_chains=num_chains,
            num_draws=num_draws,
        ),
        sample_stats=_sample_stats_group(result, num_chains=num_chains, num_draws=num_draws),
        observed_data=_observed_data_group(
            bound,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
        ),
        constant_data=_constant_data_group(
            bound,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
        ),
        coords=_freeze_coords(coords),
    )


def _posterior_group(
    bound: BoundModel,
    result: SamplerResult,
    *,
    coords: dict[str, tuple[CoordValue, ...]],
    dim_sizes: dict[str, int],
    reserved_dims: set[str],
    num_chains: int,
    num_draws: int,
) -> InferenceDataGroup:
    variables: dict[str, InferenceDataVariable] = {}
    for name, values in result.samples.items():
        array = jnp.asarray(values)
        if len(array.shape) < 2:
            raise ValueError(
                f"Posterior variable {name!r} must have shape (chain, draw, ...), got {array.shape}"
            )
        if array.shape[:2] != (num_chains, num_draws):
            raise ValueError(
                f"Posterior variable {name!r} leading sample axes must be "
                f"{(num_chains, num_draws)}, got {array.shape[:2]}"
            )
        parameter_shape = tuple(array.shape[2:])
        expected_shape = bound.param_shapes[name]
        if parameter_shape != expected_shape:
            raise ValueError(
                f"Posterior variable {name!r} has parameter shape {parameter_shape}, "
                f"but bound model expects {expected_shape}"
            )
        value_dims = _value_dims(
            name,
            parameter_shape,
            bound.dimensions,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
        )
        variables[name] = InferenceDataVariable(
            dims=(_CHAIN_DIM, _DRAW_DIM, *value_dims),
            values=array,
        )
    return _group(variables)


def _sample_stats_group(
    result: SamplerResult,
    *,
    num_chains: int,
    num_draws: int,
) -> InferenceDataGroup:
    trace = result.diagnostics.sampling
    dims = (_CHAIN_DIM, _DRAW_DIM)
    variables = {
        "diverging": InferenceDataVariable(
            dims=dims,
            values=_sample_stat_array(
                trace.is_divergent,
                name="diverging",
                num_chains=num_chains,
                num_draws=num_draws,
            ),
        ),
        "acceptance_rate": InferenceDataVariable(
            dims=dims,
            values=_sample_stat_array(
                trace.acceptance_rate,
                name="acceptance_rate",
                num_chains=num_chains,
                num_draws=num_draws,
            ),
        ),
        "n_steps": InferenceDataVariable(
            dims=dims,
            values=_sample_stat_array(
                trace.num_integration_steps,
                name="n_steps",
                num_chains=num_chains,
                num_draws=num_draws,
            ),
        ),
        "tree_depth": InferenceDataVariable(
            dims=dims,
            values=_sample_stat_array(
                trace.num_trajectory_expansions,
                name="tree_depth",
                num_chains=num_chains,
                num_draws=num_draws,
            ),
        ),
        "energy": InferenceDataVariable(
            dims=dims,
            values=_sample_stat_array(
                trace.energy,
                name="energy",
                num_chains=num_chains,
                num_draws=num_draws,
            ),
        ),
    }
    return _group(variables)


def _observed_data_group(
    bound: BoundModel,
    *,
    coords: dict[str, tuple[CoordValue, ...]],
    dim_sizes: dict[str, int],
    reserved_dims: set[str],
) -> InferenceDataGroup:
    variables: dict[str, InferenceDataVariable] = {}
    for observed in bound.meta.observed_nodes:
        array = jnp.asarray(bound.data[observed.name])
        variables[observed.name] = InferenceDataVariable(
            dims=_value_dims(
                observed.name,
                array.shape,
                bound.dimensions,
                coords=coords,
                dim_sizes=dim_sizes,
                reserved_dims=reserved_dims,
            ),
            values=array,
        )
    return _group(variables)


def _constant_data_group(
    bound: BoundModel,
    *,
    coords: dict[str, tuple[CoordValue, ...]],
    dim_sizes: dict[str, int],
    reserved_dims: set[str],
) -> InferenceDataGroup:
    variables: dict[str, InferenceDataVariable] = {}
    for name in bound.meta.data:
        array = jnp.asarray(bound.data[name])
        variables[name] = InferenceDataVariable(
            dims=_value_dims(
                name,
                array.shape,
                bound.dimensions,
                coords=coords,
                dim_sizes=dim_sizes,
                reserved_dims=reserved_dims,
            ),
            values=array,
        )
    return _group(variables)


def _value_dims(
    variable_name: str,
    shape: tuple[int, ...],
    dimensions: ResolvedModelDimensions | None,
    *,
    coords: dict[str, tuple[CoordValue, ...]],
    dim_sizes: dict[str, int],
    reserved_dims: set[str],
) -> tuple[str, ...]:
    declared = None if dimensions is None else dimensions.variables.get(variable_name)
    if declared is not None:
        if len(declared.names) != len(shape):
            raise ValueError(
                f"Dimension metadata for variable {variable_name!r} has rank "
                f"{len(declared.names)}, but value has rank {len(shape)}"
            )
        _record_dim_sizes(declared.names, shape, coords=coords, dim_sizes=dim_sizes)
        return declared.names

    fallback = tuple(
        _fallback_dim_name(
            variable_name,
            axis,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
        )
        for axis in range(len(shape))
    )
    for dim_name, axis_size in zip(fallback, shape, strict=True):
        _add_coord(coords, dim_name, tuple(range(axis_size)))
    _record_dim_sizes(fallback, shape, coords=coords, dim_sizes=dim_sizes)
    return fallback


def _validate_posterior_sample_keys(bound: BoundModel, result: SamplerResult) -> None:
    expected = set(bound.param_shapes)
    actual = set(result.samples)
    missing = expected - actual
    unexpected = actual - expected
    if missing:
        raise ValueError(f"Missing posterior samples: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected posterior samples: {sorted(unexpected)}")


def _reserved_dimension_names(dimensions: ResolvedModelDimensions | None) -> set[str]:
    if dimensions is None:
        return set()
    names: set[str] = set()
    for variable_dims in dimensions.variables.values():
        names.update(variable_dims.names)
    return names


def _validate_declared_value_dimension_names(dimensions: ResolvedModelDimensions | None) -> None:
    if dimensions is None:
        return
    reserved = {_CHAIN_DIM, _DRAW_DIM}
    for variable_name, variable_dims in dimensions.variables.items():
        conflicts = reserved.intersection(variable_dims.names)
        if conflicts:
            raise ValueError(
                f"Variable {variable_name!r} uses reserved InferenceData dimension "
                f"name(s): {sorted(conflicts)}"
            )


def _fallback_dim_name(
    variable_name: str,
    axis: int,
    *,
    coords: Mapping[str, tuple[CoordValue, ...]],
    dim_sizes: Mapping[str, int],
    reserved_dims: set[str],
) -> str:
    base = f"{variable_name}_dim_{axis}"
    if not _dim_name_is_used(base, coords=coords, dim_sizes=dim_sizes, reserved_dims=reserved_dims):
        return base
    suffix = 1
    while True:
        candidate = f"{base}_fallback_{suffix}"
        if not _dim_name_is_used(
            candidate,
            coords=coords,
            dim_sizes=dim_sizes,
            reserved_dims=reserved_dims,
        ):
            return candidate
        suffix += 1


def _dim_name_is_used(
    name: str,
    *,
    coords: Mapping[str, tuple[CoordValue, ...]],
    dim_sizes: Mapping[str, int],
    reserved_dims: set[str],
) -> bool:
    return name in coords or name in dim_sizes or name in reserved_dims


def _add_declared_coords(
    coords: dict[str, tuple[CoordValue, ...]],
    dimensions: ResolvedModelDimensions | None,
) -> None:
    if dimensions is None:
        return
    for dim_name, values in dimensions.coords.items():
        _add_coord(coords, dim_name, values)


def _add_coord(
    coords: dict[str, tuple[CoordValue, ...]],
    dim_name: str,
    values: tuple[CoordValue, ...],
) -> None:
    existing = coords.get(dim_name)
    if existing is not None and existing != values:
        raise ValueError(f"Coordinate values for dimension {dim_name!r} are inconsistent")
    coords[dim_name] = values


def _record_dim_sizes(
    dim_names: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    coords: Mapping[str, tuple[CoordValue, ...]],
    dim_sizes: dict[str, int],
) -> None:
    for dim_name, axis_size in zip(dim_names, shape, strict=True):
        existing_size = dim_sizes.get(dim_name)
        if existing_size is not None and existing_size != axis_size:
            raise ValueError(
                f"Dimension {dim_name!r} has inconsistent sizes: {existing_size} and {axis_size}"
            )
        dim_sizes[dim_name] = axis_size
        coord_values = coords.get(dim_name)
        if coord_values is not None and len(coord_values) != axis_size:
            raise ValueError(
                f"Coordinate values for dimension {dim_name!r} have length "
                f"{len(coord_values)}, but axis size is {axis_size}"
            )


def _chain_draw_shape(values: jax.Array, *, label: str) -> tuple[int, int]:
    array = jnp.asarray(values)
    if len(array.shape) != 2:
        raise ValueError(f"{label} must have shape (chain, draw), got {array.shape}")
    return (array.shape[0], array.shape[1])


def _sample_stat_array(
    values: jax.Array,
    *,
    name: str,
    num_chains: int,
    num_draws: int,
) -> jax.Array:
    array = jnp.asarray(values)
    if array.shape != (num_chains, num_draws):
        raise ValueError(
            f"sample_stats.{name} must have shape {(num_chains, num_draws)}, got {array.shape}"
        )
    return array


def _group(variables: Mapping[str, InferenceDataVariable]) -> InferenceDataGroup:
    return InferenceDataGroup(
        variables=cast(
            Mapping[str, InferenceDataVariable],
            MappingProxyType(dict(variables)),
        )
    )


def _freeze_coords(
    coords: Mapping[str, tuple[CoordValue, ...]],
) -> Mapping[str, tuple[CoordValue, ...]]:
    return cast(Mapping[str, tuple[CoordValue, ...]], MappingProxyType(dict(coords)))
