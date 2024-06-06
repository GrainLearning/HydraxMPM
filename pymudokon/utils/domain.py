"""Helper functions to discretize the domain."""


from typing import Tuple

from ..shapefunctions.shapefunction import ShapeFunction
from ..core.nodes import Nodes
from ..core.particles import Particles


def discretize(
    particles: Particles, nodes: Nodes, shapefunction: ShapeFunction, ppc: int = 2
) -> Tuple[Particles, Nodes, ShapeFunction]:
    """Discretize the domain.

    Args:
        particles (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunction (Interactions): Shape functions in the simulation.
        ppc (int, optional): Particles per cell. Defaults to 2.

    Returns:
        Tuple[Particles, Nodes, ShapeFunction]: Discretized particles, nodes and shapefunctions.
    """
    particles = particles.calculate_volume(nodes.node_spacing, ppc)

    # TODO make this a function within particles dataclass
    particles = particles.replace(masses=particles.original_density * particles.volumes)

    nodes = shapefunction.set_boundary_nodes(nodes)

    shapefunction = shapefunction.get_interactions(particles, nodes)

    shapefunction = shapefunction.calculate_shapefunction(nodes)

    return particles, nodes, shapefunction
