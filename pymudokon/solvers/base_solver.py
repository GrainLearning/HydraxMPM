import dataclasses
from typing import Callable, List
from typing_extensions import Self

import jax
import jax.numpy as jnp

from ..core.interactions import (
    Interactions,
)
from ..core.base import Base
from ..core.nodes import Nodes
from ..core.particles import Particles
from ..material.base_mat import BaseMaterial
from ..shapefunctions.base_shp import BaseShapeFunction

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class BaseSolver(Base):
    particles: Particles
    nodes: Nodes
    shapefunctions: BaseShapeFunction
    materials: List[BaseMaterial]
    forces: List[int] # TODO
    interactions: Interactions
    dt: jnp.float32

    def solve(
        self: Self,
        num_steps: jnp.int32,
        output_step: jnp.int32 = 1,
        output_function: Callable = lambda x: x,
    ):
        for step in range(num_steps):
            self = self.update()
            if step % output_step == 0:
                jax.debug.callback(output_function, (self, step))

        return self