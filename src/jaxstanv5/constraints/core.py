"""Core constraint metadata protocols and value aliases."""

from typing import Protocol

type ConstrainedValue = object
type UnconstrainedValue = object
type LogAbsDetJacobian = object


class Constraint(Protocol):
    """Bijective transform between constrained and unconstrained spaces."""

    def transform(self, x: ConstrainedValue) -> UnconstrainedValue:
        """Map constrained values to unconstrained values."""
        ...

    def inverse_transform(self, y: UnconstrainedValue) -> ConstrainedValue:
        """Map unconstrained values to constrained values."""
        ...

    def log_abs_det_jacobian(self, y: UnconstrainedValue) -> LogAbsDetJacobian:
        """Return log absolute determinant of inverse-transform Jacobian."""
        ...
