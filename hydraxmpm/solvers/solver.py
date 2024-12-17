import equinox as eqx

from typing_extensions import Tuple

from hydraxmpm.materials.material import Material

from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles


class Solver(eqx.Module):
    config: MPMConfig = eqx.field(static=True)

    particles: Particles

    nodes: Nodes

    materials: Material

    callbacks: Tuple

    def __init__(
        self,
        config: MPMConfig,
        particles=None,
        nodes=None,
        materials=None,
        forces=None,
        callbacks=None,
    ):
        self.config = config

        if forces is None:
            forces = ()

        if materials is None:
            materials = ()

        if callbacks is None:
            callbacks = ()

        self.particles = particles

        self.nodes = nodes

        self.forces = forces

        self.materials = materials
