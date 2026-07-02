def test_package_imports() -> None:
    import jaxstanv5
    import jaxstanv5.compiler
    import jaxstanv5.diagnostics
    import jaxstanv5.inference
    import jaxstanv5.model

    # Sanity: __init__ should not leak random names
    assert hasattr(jaxstanv5, "__all__")


def test_top_level_exports_the_binding_surface() -> None:
    import jaxstanv5

    assert set(jaxstanv5.__all__) == {"BoundModel", "bind_model"}


def test_compiler_exports_compile_log_density() -> None:
    import jaxstanv5.compiler as compiler
    from jaxstanv5.compiler import compile_log_density
    from jaxstanv5.compiler.core import compile_log_density as core_compile_log_density

    assert compile_log_density is core_compile_log_density
    assert compiler.__all__ == ["compile_log_density"]
