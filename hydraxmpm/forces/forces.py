import equinox as eqx
from typing import Tuple
from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from typing_extensions import Self


class Forces(eqx.Module):
    """Force state for the material properties."""

    config: MPMConfig = eqx.field(static=True)

    # def apply_on_nodes(
    #     self: Self,
    #     particles: Particles = None,
    #     nodes: Nodes = None,
    #     step: int = 0,
    # ) -> Tuple[Nodes, Self]:
    #     return nodes, self

    def apply_on_particles(
        self: Self,
        particles: Particles = None,
        nodes: Nodes = None,
        step: int = 0,
    ) -> Tuple[Particles, Self]:
        return particles, self
