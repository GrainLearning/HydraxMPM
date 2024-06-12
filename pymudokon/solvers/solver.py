"""Base solve module."""

from typing import Callable, List

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import Self

from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from ..shapefunctions.shapefunction import ShapeFunction


@struct.dataclass
class Solver:
    """State of a solver.

    Attributes:
        particles  (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunctions (ShapeFunction): Shape functions in the simulation.
        materials (List[Material]): List of materials in the simulation.
        forces (List[int]): List of forces in the simulation.
        dt (jnp.float32): Time step of the simulation.
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
            self (Solver): Self reference
            num_steps (jnp.int32): Number of steps to solve for

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
        output_start_step: jnp.int32=0,
        output_step: jnp.int32 = 1,
        output_function: Callable = lambda x: None,
    ):
        """Call the main solve loop of a solver.

        Args:
            self (Self): Self reference
            num_steps (jnp.int32): Number of steps to solve
            output_steps (jnp.int32, optional): Number of output steps. Defaults to 1.
            output_function (_type_, optional): Callback function called for every `output_step`.
                Defaults to lambda x: x.

        Returns:
            Solver: Updated solver

        Example:
            >>> def some_callback(package):
            ...     usl, step = package
            ...     # do something with usl
            ...     # e.g., print(usl.particles.positions)
            >>> usl = usl.solve(num_steps=10, output_function=some_callback)
        """

        def body_loop(step, solver):
            solver = solver.update()

            jax.lax.cond(
                (step % output_step == 0)&(output_start_step <= step),
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
        # def split_number(n, divisor):
        #     return [divisor] * (n // divisor) + ([n % divisor] if n % divisor else [])

        # segments = split_number(num_steps, output_steps)
        # current_step = 0
        # for steps in segments:
        #     self = self.solve_n(steps)
        #     current_step = current_step + steps
        #     self = output_function((current_step, self))
        return solver
