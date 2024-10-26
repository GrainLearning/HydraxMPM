"""State and functions for the background MPM grid nodes."""
# TODO: Add support for Sparse grid. This feature is currently experimental in JAX.

from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp
from jax.sharding import Sharding
import equinox as eqx


from ..config.mpm_config import MPMConfig
from pymudokon.config import mpm_config


class Nodes(eqx.Module):


    mass_stack: chex.Array
    moment_stack: chex.Array
    moment_nt_stack: chex.Array

    dim: int = eqx.field(static=True,converter=lambda x: int(x))
    num_cells: int = eqx.field(static=True,converter=lambda x: int(x))
    small_mass_cutoff: int = eqx.field(static=True,converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig=None,
        num_cells: int =None,
        dim: int= None,
        small_mass_cutoff: float = 1e-12,
    ) -> Self:
        
        if config:
             num_cells = config.num_cells
             dim = config.dim
  
        self.mass_stack=jnp.zeros((num_cells))
        self.moment_stack=jnp.zeros((num_cells, dim))
        self.moment_nt_stack=jnp.zeros((num_cells, dim))

        self.small_mass_cutoff = small_mass_cutoff
        self.dim = dim
        self.num_cells = num_cells

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
                self.moment_nt_stack.at[:].set(0.0)              
            )
        )
    # def distributed(self: Self, device: Sharding):    
    #     mass_stack = jax.device_put(self.mass_stack,device)
    #     moment_stack = jax.device_put(self.moment_stack,device)
    #     moment_nt_stack = jax.device_put(self.moment_nt_stack,device)
    #     species_stack = jax.device_put(self.species_stack,device)

    #     return self.replace(
    #         mass_stack = mass_stack,
    #         moment_stack=moment_stack,
    #         moment_nt_stack= moment_nt_stack,
    #         species_stack = species_stack
    #     )

 