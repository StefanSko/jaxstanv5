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
    log_prob = compile_log_density(bound)
    n_params = bound.n_params

    if n_params == 0:
        return SamplerResult(samples={})

    key = jax.random.PRNGKey(seed)
    init_q = jnp.zeros(n_params)

    # --- warmup ---------------------------------------------------------------
    warmup = blackjax.window_adaptation(
        blackjax.nuts,
        log_prob,
        is_mass_matrix_diagonal=True,
    )
    run_fn = cast(_WindowAdaptationRun, warmup.run)
    (last_state, tuned_params), _ = run_fn(key, init_q, num_steps=num_warmup)

    # --- sampling -------------------------------------------------------------
    sampler = blackjax.nuts(
        log_prob,
        step_size=tuned_params["step_size"],
        inverse_mass_matrix=tuned_params["inverse_mass_matrix"],
    )

    key, sample_key = jax.random.split(key)
    sample_keys = jax.random.split(sample_key, num_samples)

    @jax.jit
    def step(state: object, key: jax.Array) -> tuple[object, jax.Array]:
        new_state, _ = sampler.step(key, state)
        return new_state, new_state.position

    _, positions = jax.lax.scan(step, last_state, sample_keys)

    # positions: (num_samples, n_params)
    named = unflatten_samples(positions, bound.param_shapes)
    return SamplerResult(samples=named)
