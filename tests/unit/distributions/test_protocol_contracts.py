"""Distribution package contract tests."""

from jaxstanv5.distributions import (
    Beta,
    BetaBinomial,
    Binomial,
    Exponential,
    HalfNormal,
    MultivariateNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    Uniform,
)


def test_distribution_classes_are_available_from_package_root() -> None:
    exported: tuple[type[object], ...] = (
        Beta,
        BetaBinomial,
        Binomial,
        Exponential,
        HalfNormal,
        MultivariateNormal,
        NegativeBinomial,
        Normal,
        Poisson,
        StudentT,
        Uniform,
    )

    assert [dist.__name__ for dist in exported] == [
        "Beta",
        "BetaBinomial",
        "Binomial",
        "Exponential",
        "HalfNormal",
        "MultivariateNormal",
        "NegativeBinomial",
        "Normal",
        "Poisson",
        "StudentT",
        "Uniform",
    ]
