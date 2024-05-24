"""Helper functions to discretize the domain."""


from typing import Tuple

from ..core.interactions import Interactions
from ..core.nodes import Nodes
from ..core.particles import Particles


def discretize(
    particles: Particles, nodes: Nodes, shapefunctions: Interactions, ppc: int = 2
) -> Tuple[Particles, Nodes, Interactions]:
    """Discretize the domain.

    Args:
        particles (Particles): Particles in the simulation.
        nodes (Nodes): Nodes in the simulation.
        shapefunctions (Interactions): Shape functions in the simulation.
        ppc (int, optional): Particles per cell. Defaults to 2.

    Returns:
        Interactions: Interactions object with discretized domain.
    """
    particles = particles.calculate_volume(nodes.node_spacing, ppc)

    # TODO make this a function within particles dataclass
    particles = particles.replace(masses=particles.original_density * particles.volumes)

    nodes = shapefunctions.set_boundary_nodes(nodes)

    shapefunctions = shapefunctions.get_interactions(particles, nodes)

    shapefunctions = shapefunctions.calculate_shapefunction(nodes)

    return particles, nodes, shapefunctions
