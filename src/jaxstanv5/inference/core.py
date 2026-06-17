"""Core inference — sampling via BlackJAX NUTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple, Protocol, cast

import blackjax
import jax
import jax.numpy as jnp

from jaxstanv5._backends.jax.constraints import inverse_transform
from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.model.bound import BoundModel
from jaxstanv5.model.decorator import ModelMeta, _resolved_free_values


class NutsDiagnosticTrace(NamedTuple):
    """Per-transition NUTS diagnostics.

    Arrays have shape ``(num_chains, num_steps)`` on public sampler results.
    """

    is_divergent: jax.Array
    acceptance_rate: jax.Array
    num_integration_steps: jax.Array
    num_trajectory_expansions: jax.Array
    energy: jax.Array


class SamplerDiagnostics(NamedTuple):
    """Warmup and post-warmup NUTS diagnostics."""

    warmup: NutsDiagnosticTrace
    sampling: NutsDiagnosticTrace


class _NutsInfo(Protocol):
    """Subset of BlackJAX NUTSInfo used in public diagnostics."""

    is_divergent: jax.Array
    acceptance_rate: jax.Array
    num_integration_steps: jax.Array
    num_trajectory_expansions: jax.Array
    energy: jax.Array


class _AdaptationInfo(Protocol):
    """Subset of BlackJAX warmup info used in public diagnostics."""

    info: _NutsInfo


class _SampleBlock(NamedTuple):
    """Stacked unconstrained positions and diagnostics for one chain."""

    positions: jax.Array
    diagnostics: NutsDiagnosticTrace


class _ChainSample(NamedTuple):
    """Samples and diagnostics for one or more chains."""

    samples: dict[str, jax.Array]
    diagnostics: SamplerDiagnostics


class _WindowAdaptationRun(Protocol):
    """BlackJAX window adaptation run callable.

    BlackJAX's public ``RunFn`` protocol omits the supported ``num_steps``
    argument, so keep the cast narrow at the BlackJAX boundary.
    """

    def __call__(
        self,
        rng_key: jax.Array,
        position: jax.Array,
        *,
        num_steps: int,
    ) -> tuple[tuple[object, Mapping[str, jax.Array]], _AdaptationInfo]: ...


class _DrawSamples(Protocol):
    """JIT-compiled post-warmup draw loop."""

    def __call__(
        self,
        state: object,
        sample_keys: jax.Array,
        step_size: jax.Array,
        inverse_mass_matrix: jax.Array,
    ) -> tuple[object, _SampleBlock]: ...


@dataclass(frozen=True)
class SamplerResult:
    """Results of MCMC sampling.

    Attributes
    ----------
    samples
        Parameter names to arrays of shape ``(num_chains, num_samples, *param_shape)``.
        The leading dimension is the chain.
    diagnostics
        Warmup and post-warmup NUTS diagnostics with arrays of shape
        ``(num_chains, num_steps)``.
    """

    samples: dict[str, jax.Array]
    diagnostics: SamplerDiagnostics


def _diagnostic_trace_from_nuts_info(info: _NutsInfo) -> NutsDiagnosticTrace:
    """Extract backend-neutral scalar diagnostics from BlackJAX NUTS info."""
    return NutsDiagnosticTrace(
        is_divergent=jnp.asarray(info.is_divergent),
        acceptance_rate=jnp.asarray(info.acceptance_rate),
        num_integration_steps=jnp.asarray(info.num_integration_steps),
        num_trajectory_expansions=jnp.asarray(info.num_trajectory_expansions),
        energy=jnp.asarray(info.energy),
    )


def _empty_diagnostic_trace(shape: tuple[int, int]) -> NutsDiagnosticTrace:
    """Return placeholder diagnostics for models with no NUTS transitions."""
    return NutsDiagnosticTrace(
        is_divergent=jnp.zeros(shape, dtype=bool),
        acceptance_rate=jnp.full(shape, jnp.nan),
        num_integration_steps=jnp.zeros(shape, dtype=jnp.int32),
        num_trajectory_expansions=jnp.zeros(shape, dtype=jnp.int32),
        energy=jnp.full(shape, jnp.nan),
    )


def _empty_sampler_diagnostics(
    *,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
) -> SamplerDiagnostics:
    """Return placeholder diagnostics for models with no NUTS coordinates."""
    return SamplerDiagnostics(
        warmup=_empty_diagnostic_trace((num_chains, num_warmup)),
        sampling=_empty_diagnostic_trace((num_chains, num_samples)),
    )


def _validate_positive_count(value: int, *, name: str) -> None:
    """Validate one public sampler count before entering JAX/BlackJAX."""
    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def _validate_sampler_counts(
    *,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
) -> None:
    """Validate public sampler loop dimensions."""
    _validate_positive_count(num_chains, name="num_chains")
    _validate_positive_count(num_warmup, name="num_warmup")
    _validate_positive_count(num_samples, name="num_samples")


def _draw_initial_position(rng_key: jax.Array, n_params: int) -> jax.Array:
    """Draw one overdispersed unconstrained initial position."""
    return jax.random.uniform(rng_key, (n_params,), minval=-2.0, maxval=2.0)


def _empty_unconstrained_samples(
    shapes: dict[str, tuple[int, ...]],
    *,
    num_chains: int,
    num_samples: int,
) -> dict[str, jax.Array]:
    """Return empty sample arrays for declared zero-sized free values."""
    return {name: jnp.zeros((num_chains, num_samples, *shape)) for name, shape in shapes.items()}


def _unflatten_samples(
    flat: jax.Array,
    shapes: dict[str, tuple[int, ...]],
) -> dict[str, jax.Array]:
    """Split flat (num_samples, n_params) into named arrays with chain dim.

    Returns arrays of shape ``(1, num_samples, *param_shape)``.
    """
    result: dict[str, jax.Array] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 1
        for d in shape:
            size *= d
        if shape == ():
            param = flat[:, offset]
        else:
            param = flat[:, offset : offset + size].reshape(flat.shape[0], *shape)
        # Add chain dimension: (N,) → (1, N), (N, *shape) → (1, N, *shape)
        result[name] = jnp.expand_dims(param, axis=0)
        offset += size
    return result


def _constrain_sample_values(
    samples: dict[str, jax.Array],
    meta: ModelMeta,
) -> dict[str, jax.Array]:
    """Map sampled unconstrained parameter values back to constrained values."""
    result: dict[str, jax.Array] = {}
    free_values = _resolved_free_values(meta)
    for name, values in samples.items():
        constraint = free_values[name].constraint
        if constraint is None:
            result[name] = values
        else:
            result[name] = jnp.asarray(inverse_transform(constraint, values))
    return result


def _sample_one_chain(
    bound: BoundModel,
    warmup_run: _WindowAdaptationRun,
    draw_samples: _DrawSamples,
    rng_key: jax.Array,
    *,
    num_warmup: int,
    num_samples: int,
) -> _ChainSample:
    """Draw one NUTS chain for a bound model.

    Sample arrays have shape ``(1, num_samples, *param_shape)``.  The single
    leading chain axis keeps the one-chain path compatible with stacked
    multi-chain results.
    """
    init_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    init_q = _draw_initial_position(init_key, bound.n_params)

    (last_state, tuned_params), warmup_info = warmup_run(
        warmup_key,
        init_q,
        num_steps=num_warmup,
    )
    warmup_diagnostics = _diagnostic_trace_from_nuts_info(warmup_info.info)

    sample_keys = jax.random.split(sample_key, num_samples)
    _, sample_block = draw_samples(
        last_state,
        sample_keys,
        tuned_params["step_size"],
        tuned_params["inverse_mass_matrix"],
    )

    unconstrained = _unflatten_samples(sample_block.positions, bound.param_shapes)
    return _ChainSample(
        samples=_constrain_sample_values(unconstrained, bound.meta),
        diagnostics=SamplerDiagnostics(
            warmup=warmup_diagnostics,
            sampling=sample_block.diagnostics,
        ),
    )


def _sample_chains(
    bound: BoundModel,
    warmup_run: _WindowAdaptationRun,
    draw_samples: _DrawSamples,
    rng_key: jax.Array,
    *,
    num_chains: int,
    num_warmup: int,
    num_samples: int,
) -> _ChainSample:
    """Draw independent chains by batching the one-chain sampler."""
    chain_keys = jax.random.split(rng_key, num_chains)

    def sample_chain(chain_key: jax.Array) -> _ChainSample:
        return _sample_one_chain(
            bound,
            warmup_run,
            draw_samples,
            chain_key,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )

    batched = jax.vmap(sample_chain)(chain_keys)
    return _ChainSample(
        samples={name: jnp.squeeze(values, axis=1) for name, values in batched.samples.items()},
        diagnostics=batched.diagnostics,
    )


@dataclass(frozen=True, init=False)
class CompiledSampler:
    """NUTS sampler compiled for one concrete bound model.

    Bound data is closed over by the compiled log density, so this object is
    reusable only for repeated runs of the same ``BoundModel`` data. Distinct
    ``num_warmup``, ``num_samples``, or ``num_chains`` values may trigger
    additional backend compilation.
    """

    _bound: BoundModel
    _warmup_run: _WindowAdaptationRun
    _draw_samples: _DrawSamples

    def __init__(self) -> None:
        raise TypeError("Use compile_sampler(...) to create a CompiledSampler")

    @classmethod
    def _create(
        cls,
        bound: BoundModel,
        warmup_run: _WindowAdaptationRun,
        draw_samples: _DrawSamples,
    ) -> CompiledSampler:
        sampler = cast(CompiledSampler, object.__new__(cls))
        object.__setattr__(sampler, "_bound", bound)
        object.__setattr__(sampler, "_warmup_run", warmup_run)
        object.__setattr__(sampler, "_draw_samples", draw_samples)
        return sampler

    def sample(
        self,
        seed: int,
        num_warmup: int,
        num_samples: int,
        *,
        num_chains: int = 1,
    ) -> SamplerResult:
        """Draw posterior samples with the compiled sampler."""
        _validate_sampler_counts(
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        if self._bound.n_params == 0:
            unconstrained = _empty_unconstrained_samples(
                self._bound.param_shapes,
                num_chains=num_chains,
                num_samples=num_samples,
            )
            return SamplerResult(
                samples=_constrain_sample_values(unconstrained, self._bound.meta),
                diagnostics=_empty_sampler_diagnostics(
                    num_chains=num_chains,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                ),
            )

        key = jax.random.PRNGKey(seed)
        chain_sample = _sample_chains(
            self._bound,
            self._warmup_run,
            self._draw_samples,
            key,
            num_chains=num_chains,
            num_warmup=num_warmup,
            num_samples=num_samples,
        )
        return SamplerResult(samples=chain_sample.samples, diagnostics=chain_sample.diagnostics)


def _validate_target_acceptance_rate(target_acceptance_rate: float) -> None:
    """Validate the NUTS adaptation target acceptance rate."""
    if not 0.0 < target_acceptance_rate < 1.0:
        raise ValueError("target_acceptance_rate must be in (0, 1)")


def compile_sampler(
    bound: BoundModel,
    *,
    target_acceptance_rate: float = 0.8,
) -> CompiledSampler:
    """Compile NUTS machinery for one concrete bound model.

    The bound model's data is part of the compiled log-density closure. Reusing
    the returned sampler avoids rebuilding the Python sampler wrapper, but new
    loop sizes may still retrace/recompile backend executables.
    """
    _validate_target_acceptance_rate(target_acceptance_rate)
    if bound.n_params == 0:

        def warmup_run(
            rng_key: jax.Array,
            position: jax.Array,
            *,
            num_steps: int,
        ) -> tuple[tuple[object, Mapping[str, jax.Array]], _AdaptationInfo]:
            raise RuntimeError("Parameterless models do not run warmup")

        def _draw_samples(
            state: object,
            sample_keys: jax.Array,
            step_size: jax.Array,
            inverse_mass_matrix: jax.Array,
        ) -> tuple[object, _SampleBlock]:
            raise RuntimeError("Parameterless models do not draw samples")

        return CompiledSampler._create(
            bound=bound, warmup_run=warmup_run, draw_samples=_draw_samples
        )

    log_prob = compile_log_density(bound)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob,
        is_mass_matrix_diagonal=True,
        target_acceptance_rate=target_acceptance_rate,
    )
    warmup_run = cast(_WindowAdaptationRun, warmup.run)
    nuts_kernel = blackjax.nuts.build_kernel()

    @jax.jit
    def draw_samples(
        state: object,
        sample_keys: jax.Array,
        step_size: jax.Array,
        inverse_mass_matrix: jax.Array,
    ) -> tuple[object, _SampleBlock]:
        def step(carry: object, key: jax.Array) -> tuple[object, _SampleBlock]:
            new_state, info = nuts_kernel(
                key,
                carry,
                log_prob,
                step_size,
                inverse_mass_matrix,
            )
            diagnostics = _diagnostic_trace_from_nuts_info(cast(_NutsInfo, info))
            return new_state, _SampleBlock(
                positions=new_state.position,
                diagnostics=diagnostics,
            )

        return jax.lax.scan(step, state, sample_keys)

    return CompiledSampler._create(
        bound=bound,
        warmup_run=warmup_run,
        draw_samples=cast(_DrawSamples, draw_samples),
    )


def sample(
    bound: BoundModel,
    seed: int,
    num_warmup: int,
    num_samples: int,
    *,
    num_chains: int = 1,
    target_acceptance_rate: float = 0.8,
) -> SamplerResult:
    """Draw posterior samples via NUTS with window adaptation.

    Parameters
    ----------
    bound
        Bound model with concrete data.
    seed
        PRNG seed.
    num_warmup
        Number of warmup / adaptation steps.
    num_samples
        Number of post-warmup draws per chain.
    num_chains
        Number of independent NUTS chains to run.
    target_acceptance_rate
        Target acceptance probability used during NUTS step-size adaptation.

    Returns
    -------
    SamplerResult
        Drawn samples per parameter.
    """
    _validate_sampler_counts(
        num_chains=num_chains,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
    return compile_sampler(bound, target_acceptance_rate=target_acceptance_rate).sample(
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
    )
