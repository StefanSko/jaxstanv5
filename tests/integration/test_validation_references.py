"""Tests for analytic posterior references used by validation harnesses."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from _validation import (
    hierarchical_normal_known_scale_reference,
    normal_known_scale_reference,
)


def test_normal_known_scale_reference_returns_conjugate_posterior() -> None:
    reference = normal_known_scale_reference(
        parameter="mu",
        y=jnp.array([1.0, 2.0, 3.0]),
        prior_loc=0.0,
        prior_scale=2.0,
        obs_scale=1.0,
    )

    assert reference.parameter == "mu"
    assert reference.mean == pytest.approx(1.8461538462)
    assert reference.sd == pytest.approx(0.5547001962)


def test_hierarchical_normal_known_scale_reference_returns_gaussian_posterior() -> None:
    reference = hierarchical_normal_known_scale_reference(
        population_parameter="mu_pop",
        group_parameter="theta",
        y=jnp.array([[1.0, 1.5], [2.0, 2.5]]),
        prior_loc=0.0,
        prior_scale=2.0,
        group_scale=1.0,
        obs_scale=0.5,
    )

    assert reference.population.parameter == "mu_pop"
    assert reference.population.mean == pytest.approx(1.5342465753)
    assert reference.population.sd == pytest.approx(0.7022468832)

    assert tuple(group.parameter for group in reference.groups) == ("theta[0]", "theta[1]")
    assert tuple(group.mean for group in reference.groups) == pytest.approx(
        (1.2815821918, 2.1704719067)
    )
    assert tuple(group.sd for group in reference.groups) == pytest.approx(
        (0.3423439562, 0.3423439562)
    )
