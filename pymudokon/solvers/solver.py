"""Base solve module."""

from typing import Callable, List

import jax
import jax.numpy as jnp
import chex
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass
class Solver:
    """Base solver class.

    A top level wrapper to manage the solver state make impure callback functions.

    Attributes:
        particles: Particles in the simulation.
        nodes: Nodes in the simulation.
        shapefunctions: Shape functions in the simulation.
        materials: List of materials in the simulation.
        forces: List of forces in the simulation.
        dt: Time step of the simulation.
    """

    particles: Particles
    nodes: Nodes
    materials: List[Material]
    forces: List[Forces]
    shapefunctions: ShapeFunction
    dt: jnp.float32

    def update(self: Self):
        """Update the state of the solver."""
        pass

    @jax.jit
    def solve_n(self: Self, num_steps: jnp.int32) -> Self:
        """Solve the solver for `num_steps`.

        Args:
            self: Self reference
            num_steps: Number of steps to solve for

        Returns:
            Solver: Updated solver
        """
        usl = jax.lax.fori_loop(
            0,
            num_steps,
            lambda step, usl: usl.update(),
            self,
        )
        return usl

    @jax.jit
    def solve(
        self: Self,
        num_steps: jnp.int32,
        output_start_step: jnp.int32 = 0,
        output_step: jnp.int32 = 1,
        output_function: Callable = lambda x: None,
    ):
        """Call the main solve loop of a solver.

        Args:
            self: Self reference
            num_steps: Number of steps to solve
            output_steps (optional): Number of output steps. Defaults to 1.
            output_function (optional): Callback function called for every `output_step`.
                Defaults to lambda x: x.

        Returns:
            Solver: Updated solver
        """

        def body_loop(step, solver):
            solver = solver.update()

            jax.lax.cond(
                (step % output_step == 0) & (output_start_step <= step),
                lambda x: jax.experimental.io_callback(output_function, None, x),
                lambda x: None,
                (step, solver),
            )
            return solver

        solver = jax.lax.fori_loop(
            0,
            num_steps,
            body_loop,
            self,
        )
        return solver
