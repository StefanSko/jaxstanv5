def test_package_imports() -> None:
    import jaxstanv5
    import jaxstanv5.compiler
    import jaxstanv5.constraints
    import jaxstanv5.diagnostics
    import jaxstanv5.distributions
    import jaxstanv5.inference
    import jaxstanv5.model

    # Sanity: __init__ should not leak random names
    assert hasattr(jaxstanv5, "__all__")
