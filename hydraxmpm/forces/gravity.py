"""Module for the gravity force. Impose gravity on the nodes."""

from typing import List, Optional, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..common.types import TypeFloat, TypeFloatVector, TypeInt
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .force import Force


class Gravity(Force):
    """Gravity force enforced on the background grid."""

    gravity: TypeFloatVector
    increment: Optional[TypeFloatVector]
    stop_ramp_step: Optional[TypeInt]
    particle_gravity: bool = eqx.field(static=True, converter=lambda x: bool(x))

    dt: TypeFloat = eqx.field(static=True)

    def __init__(
        self: Self,
        gravity: TypeFloatVector | List | Tuple,
        increment: Optional[TypeFloatVector | List | Tuple] = None,
        stop_ramp_step: Optional[TypeInt] = 0,
        particle_gravity: Optional[bool] = True,
        **kwargs,
    ) -> Self:
        """Initialize Gravity force on Nodes."""
        self.gravity = jnp.array(gravity)

        self.increment = (
            jnp.zeros_like(self.gravity) if increment is None else increment
        )
        self.stop_ramp_step = stop_ramp_step
        self.particle_gravity = particle_gravity

        self.dt = kwargs.get("dt", 0.001)

    def apply_on_grid(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: TypeFloat = 0.01,
        dim: TypeInt = 3,
        **kwargs: Any,
    ) -> Tuple[Grid, Self]:
        """Apply gravity on the nodes."""

        if self.particle_gravity:
            return grid, self

        if self.increment is not None:
            gravity = self.gravity + self.increment * jnp.minimum(
                step, self.stop_ramp_step
            )
        else:
            gravity = self.gravity

        moment_gravity = grid.mass_stack.reshape(-1, 1) * gravity * dt

        new_moment_nt_stack = grid.moment_nt_stack + moment_gravity
        # not used?
        # new_moment_stack = grid.moment_stack + moment_gravity

        new_grid = eqx.tree_at(
            lambda state: (state.moment_nt_stack),
            grid,
            (new_moment_nt_stack),
        )

        # self is updated if there is a gravity ramp
        return new_grid, self

    def apply_on_points(
        self: Self,
        material_points: Optional[MaterialPoints] = None,
        grid: Optional[Grid] = None,
        step: Optional[TypeInt] = 0,
        dt: Optional[TypeFloat] = 0.01,
        dim: TypeInt = 3,
    ) -> Tuple[MaterialPoints, Self]:
        if not self.particle_gravity:
            return material_points, self

        if self.increment is not None:
            gravity = self.gravity + self.increment * jnp.minimum(
                step, self.stop_ramp_step
            )
        else:
            gravity = self.gravity
        # jax.debug.print("gravity {}", gravity)

        def get_gravitational_force(mass: TypeFloat):
            return mass * gravity

        new_particles = eqx.tree_at(
            lambda state: (state.force_stack),
            material_points,
            (jax.vmap(get_gravitational_force)(material_points.mass_stack)),
        )
        return new_particles, self
