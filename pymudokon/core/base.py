"""Base class for all pytree classes in the pymudokon library."""

import dataclasses

import jax
from typing_extensions import Self


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class Base:
    """Base class with required default pytree functions."""

    def tree_flatten(self):
        """Tree flatten the pytree nodes."""
        children = tuple(getattr(self, f.name) for f in dataclasses.fields(self))
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten the pytree nodes."""
        kwargs = {f.name: v for f, v in zip(dataclasses.fields(cls), children)}
        return cls(**kwargs)

    @jax.jit
    def replace(self, **changes) -> Self:
        """Replace the attributes of the class."""
        return dataclasses.replace(self, **changes)
