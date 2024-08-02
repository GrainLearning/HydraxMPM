"""Helper functions to discretize the domain."""

from typing import Tuple

from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction


def discretize(
    particles: Particles,
    nodes: Nodes,
    shapefunction: ShapeFunction,
    ppc: int = 2,
    density_ref: float = 1000,
) -> Tuple[Particles, Nodes, ShapeFunction]:
    """Discretize the domain.

    Args:
        particles (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunction (Interactions): Shape functions in the simulation.
        ppc (int, optional): Particles per cell. Defaults to 2.
        density_ref (float, optional): Reference density. Defaults to 1000.

    Returns:
        Tuple[Particles, Nodes, ShapeFunction]: Discretized particles,
        nodes and shapefunctions.
    """
    particles = particles.calculate_volume(nodes.node_spacing, ppc)

    # TODO make reference density an array (possible)
    # TODO make this a function within particles dataclass
    particles = particles.replace(mass_stack=density_ref * particles.volume_stack)

    return particles, nodes, shapefunction
