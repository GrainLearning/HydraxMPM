"""Module containing base forces state."""


from typing_extensions import Self

import chex


# from ..core.nodes import Nodes
# from ..core.particles import Particles
# from ..shapefunctions.shapefunction import ShapeFunction


@chex.dataclass
class Forces:
    """Force state for the material properties."""

    @classmethod
    def create(cls: Self) -> Self:
        """Initialize the force state."""
        return cls()
