import equinox as eqx
import jax.numpy as jnp
from typing_extensions import Self

from ..common.types import TypeFloatScalarNStack, TypeFloatVectorNStack
from ..config.mpm_config import MPMConfig
from .grid import Grid


class Nodes(Grid):
    """Represents the background grid nodes in an MPM simulation.

    These nodes inherit mapping functionalities from the :class:`Grid` class
    (TODO: Add link to Grid class).

    This class is typically initialized internally by the solver
    but can be instantiated directly if needed.


    Attributes:
        mass_stack (jnp.ndarray): Mass assigned to each grid node (shape: `(num_nodes)`).
        moment_stack (jnp.ndarray): Momentum (velocity * mass) stored at each grid node (shape: `(num_nodes, dim)`).
        moment_nt_stack (jnp.ndarray): Momentum at the next time step, used for integration schemes like FLIP (shape: `(num_nodes, dim)`).
        normal_stack (jnp.ndarray): Normal vectors associated with each node (shape: `(num_nodes, dim)`).
            This might represent surface normals if the grid represents a boundary.
        small_mass_cutoff (float): Threshold for small mass values.
            Nodes with mass below this cutoff may be treated specially to avoid numerical instabilities.

    """

    mass_stack: TypeFloatScalarNStack
    moment_stack: TypeFloatVectorNStack
    moment_nt_stack: TypeFloatVectorNStack
    normal_stack: TypeFloatVectorNStack
    small_mass_cutoff: float = eqx.field(static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig = None,
        small_mass_cutoff: float = 1e-10,
    ) -> Self:
        self.mass_stack = jnp.zeros((config.num_cells), device=config.device)
        self.moment_stack = jnp.zeros(
            (config.num_cells, config.dim), device=config.device
        )
        self.moment_nt_stack = jnp.zeros(
            (config.num_cells, config.dim), device=config.device
        )
        self.normal_stack = jnp.zeros(
            (config.num_cells, config.dim), device=config.device
        )
        self.small_mass_cutoff = small_mass_cutoff
        super().__init__(config)

    def refresh(self: Self) -> Self:
        """Reset background MPM node states."""

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
