from typing import List, Optional, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..common.types import TypeFloat, TypeFloatVector, TypeInt
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .force import Force


class ParticleDamping(Force):
    alpha: TypeFloat
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def apply_on_points(
      self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
    ) -> Tuple[MaterialPoints, Self]:
        """
        Apply velocity-proportional damping to material points.

        Args:
            material_points: The particles to apply damping to.
            grid: Not used here.
            step: Current simulation step (unused).
            dt: Time step size (unused).
            dim: Problem dimension (unused).

        Returns:
            Updated material points and (possibly updated) ParticleDamping instance.
        """
        if material_points is None:
            return material_points, self

        # Damping force: F = -alpha * v
        damping_force_stack = material_points.force_stack -self.alpha * material_points.velocity_stack
        new_particles = eqx.tree_at(
            lambda state: state.force_stack,
            material_points,
            (damping_force_stack),
        )
        return new_particles, self

