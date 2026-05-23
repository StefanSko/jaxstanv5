"""Core inference — sampling via BlackJAX NUTS."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import blackjax
import jax
import jax.numpy as jnp

from jaxstanv5.compiler.core import compile_log_density
from jaxstanv5.model.bound import BoundModel


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
    ) -> tuple[tuple[object, Mapping[str, jax.Array]], object]: ...


class _DrawSamples(Protocol):
    """JIT-compiled post-warmup draw loop."""

    def __call__(
        self,
        state: object,
        sample_keys: jax.Array,
        step_size: jax.Array,
        inverse_mass_matrix: jax.Array,
    ) -> tuple[object, jax.Array]: ...


@dataclass(frozen=True)
class SamplerResult:
    """Results of MCMC sampling.

    Attributes
    ----------
    samples
        Parameter names to arrays of shape ``(1, num_samples, *param_shape)``.
        The leading dimension is the chain (always 1 for single-chain NUTS).
    """

    samples: dict[str, jax.Array]


def unflatten_samples(
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


@dataclass(frozen=True, init=False)
class CompiledSampler:
    """Reusable NUTS sampler for one bound model shape.

    Compile once for repeated same-shape sampling runs.  The simple ``sample``
    function remains the one-shot convenience path.
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
    ) -> SamplerResult:
        """Draw posterior samples with the compiled sampler."""
        if self._bound.n_params == 0:
            return SamplerResult(samples={})

        key = jax.random.PRNGKey(seed)
        init_q = jnp.zeros(self._bound.n_params)

        (last_state, tuned_params), _ = self._warmup_run(key, init_q, num_steps=num_warmup)

        key, sample_key = jax.random.split(key)
        sample_keys = jax.random.split(sample_key, num_samples)
        _, positions = self._draw_samples(
            last_state,
            sample_keys,
            tuned_params["step_size"],
            tuned_params["inverse_mass_matrix"],
        )

        return SamplerResult(samples=unflatten_samples(positions, self._bound.param_shapes))


def compile_sampler(bound: BoundModel) -> CompiledSampler:
    """Compile a reusable NUTS sampler for one bound model shape."""
    if bound.n_params == 0:

        def warmup_run(
            rng_key: jax.Array,
            position: jax.Array,
            *,
            num_steps: int,
        ) -> tuple[tuple[object, Mapping[str, jax.Array]], object]:
            raise RuntimeError("Parameterless models do not run warmup")

        def draw_samples(
            state: object,
            sample_keys: jax.Array,
            step_size: jax.Array,
            inverse_mass_matrix: jax.Array,
        ) -> tuple[object, jax.Array]:
            raise RuntimeError("Parameterless models do not draw samples")

        return CompiledSampler._create(
            bound=bound, warmup_run=warmup_run, draw_samples=draw_samples
        )

    log_prob = compile_log_density(bound)
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob,
        is_mass_matrix_diagonal=True,
    )
    warmup_run = cast(_WindowAdaptationRun, warmup.run)
    nuts_kernel = blackjax.nuts.build_kernel()

    @jax.jit
    def draw_samples(
        state: object,
        sample_keys: jax.Array,
        step_size: jax.Array,
        inverse_mass_matrix: jax.Array,
    ) -> tuple[object, jax.Array]:
        def step(carry: object, key: jax.Array) -> tuple[object, jax.Array]:
            new_state, _ = nuts_kernel(
                key,
                carry,
                log_prob,
                step_size,
                inverse_mass_matrix,
            )
            return new_state, new_state.position

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
        Number of post-warmup draws.

    Returns
    -------
    SamplerResult
        Drawn samples per parameter.
    """
    return compile_sampler(bound).sample(
        seed=seed,
        num_warmup=num_warmup,
        num_samples=num_samples,
    )
