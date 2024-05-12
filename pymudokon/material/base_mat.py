"""Module for containing base class for the material."""

import dataclasses

import jax

from ..core.base import Base


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class BaseMaterial(Base):
    """BaseMaterial state for the material properties."""
