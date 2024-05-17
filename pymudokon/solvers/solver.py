"""Base solve module."""

import dataclasses
from typing import Callable, List, Dict

import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..core.base import Base
from ..core.interactions import (
    Interactions,
)
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..forces.forces import Forces
from ..materials.material import Material
from ..shapefunctions.shapefunction import ShapeFunction


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Solver(Base):
    """State of a solver.

    Attributes:
        particles  (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunctions (ShapeFunction):
            Shape functions in the simulation.
        materials (List[Material]):
            List of materials in the simulation.
        forces (List[int]):
            List of forces in the simulation.
        interactions (Interactions):
            Interactions in the simulation.
        dt (jnp.float32):
            time step of the simulation.
    """

    particles: Particles
    nodes: Nodes
    shapefunctions: ShapeFunction
    materials: List[Material]
    forces: List[Forces]
    interactions: Interactions
    dt: jnp.float32

    def solve(
        self: Self,
        num_steps: jnp.int32,
        output_step: jnp.int32 = 1,
        output_function: Callable = lambda x: x,
        output_function_args: Dict = {},
    ):
        """Call the main solve loop of a solver.

        Args:
            self (Self):
                Self reference
            num_steps (jnp.int32):
                Number of steps to solve
            output_step (jnp.int32, optional):
                Number of output steps. Defaults to 1.
            output_function (_type_, optional):
                Callback function called for every `output_step`. Defaults to lambdax:x.

        Returns:
            Solver: Updated solver

        Example:
            >>> def some_callback(package):
            ...     usl, step = package
            ...     # do something with usl
            ...     # e.g., print(usl.particles.positions)
            >>> usl = usl.solve(num_steps=10, output_function=some_callback)
        """
        for step in range(num_steps):
            self = self.update()
            if step % output_step == 0:
                # jax.debug.callback(output_function, (self, step), output_function_args=output_function_args)
                output_function((self, step, *output_function_args))
        return self
