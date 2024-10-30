from typing_extensions import Self

import chex

import jax.numpy as jnp

import equinox as eqx


from ..config.mpm_config import MPMConfig

    
from .grid import Grid

class Nodes(Grid):
    mass_stack: chex.Array
    moment_stack: chex.Array
    moment_nt_stack: chex.Array
    small_mass_cutoff: int = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig = None,
        small_mass_cutoff: float = 1e-12,
    ) -> Self:


        self.mass_stack = jnp.zeros((config.num_cells))
        self.moment_stack = jnp.zeros((config.num_cells, config.dim))
        self.moment_nt_stack = jnp.zeros((config.num_cells, config.dim))
        self.small_mass_cutoff = small_mass_cutoff
        super().__init__(config)


    def refresh(self: Self) -> Self:
        """Reset background MPM node states.

        Args:
            self: Nodes state.

        Returns:
            Nodes: Updated node state.
        """

        return eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
            ),
            self,
            (
                self.mass_stack.at[:].set(0.0),
                self.moment_stack.at[:].set(0.0),
                self.moment_nt_stack.at[:].set(0.0),
            ),
        )
