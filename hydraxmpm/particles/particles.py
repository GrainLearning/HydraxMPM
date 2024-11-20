import chex
import equinox as eqx
import jax.numpy as jnp
from typing_extensions import Optional, Self
import jax
from ..config.mpm_config import MPMConfig


class Particles(eqx.Module):
    """

    Lagrangian markers that can move across the background grid.

    Attributes:
        position_stack: spatial coordinate vectors `(num_points, dimension)`.
        velocity_stack: spatial velocity vectors `(num_points, dimension)`.
        force_stack: external force vectors `(num_points, dimension)`.
        mass_stack: Masses `(num_points,)`.
        volume_stack: Current volumes  `(num_points,)`.
        volume0_stack: Original volumes `(num_points,)`.
        L_stack: Velocity gradient tensors `(num_points, 3, 3)`.
        stress_stack: Cauchy stress tensors `(num_points, 3, 3)`.
        F_stack: Deformation gradient tensors `(num_points, 3, 3)`.
        id_stack: Particle IDs `(num_points,)`.

    """

    position_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    velocity_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    force_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    mass_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    volume_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    volume0_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    L_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    stress_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))
    F_stack: chex.Array = eqx.field(converter=lambda x: jnp.asarray(x))

    config: MPMConfig = eqx.field(static=True)

    def __init__(
        self: Self,
        config: MPMConfig,
        position_stack: chex.Array = None,
        velocity_stack: Optional[chex.Array] = None,
        mass_stack: Optional[chex.Array] = None,
        volume_stack: Optional[chex.Array] = None,
        L_stack: Optional[chex.Array] = None,
        stress_stack: Optional[chex.Array] = None,
        force_stack: Optional[chex.Array] = None,
        F_stack: Optional[chex.Array] = None,
    ) -> Self:
        """
        Args:
            self (Self): class
            config (MPMConfig optional): MPM configuration.
            position_stack (chex.Array, optional): array of positions. Defaults to None.
            velocity_stack (chex.Array, optional): array of velocities. Defaults to None.
            mass_stack (chex.Array, optional): array of masses. Defaults to None.
            volume_stack (chex.Array, optional): array of volumes. Defaults to None.
            L_stack (chex.Array, optional): array of velocity gradients. Defaults to None.
            stress_stack (chex.Array, optional): array of cauchy stress tensors. Defaults to None.
            force_stack (chex.Array, optional): array of forces. Defaults to None.
            F_stack (chex.Array, optional): array of deformation gradients. Defaults to None.

        Returns:
            Self: particle class
        """
        self.config = config

        if position_stack is None:
            self.position_stack = jnp.zeros(
                (self.config.num_points, self.config.dim), device=self.config.device
            )

        else:
            self.position_stack = jax.device_put(
                position_stack, device=self.config.device
            )

        if velocity_stack is None:
            self.velocity_stack = jnp.zeros(
                (self.config.num_points, self.config.dim), device=self.config.device
            )
        else:
            self.velocity_stack = jax.device_put(
                velocity_stack, device=self.config.device
            )

        if force_stack is None:
            self.force_stack = jnp.zeros(
                (self.config.num_points, self.config.dim), device=self.config.device
            )
        else:
            self.force_stack = jax.device_put(force_stack, device=self.config.device)

        if mass_stack is None:
            self.mass_stack = jnp.zeros(
                self.config.num_points, device=self.config.device
            )
        else:
            self.mass_stack = jax.device_put(mass_stack, device=self.config.device)

        if volume_stack is None:
            self.volume_stack = jnp.zeros(
                self.config.num_points, device=self.config.device
            )
        else:
            self.volume_stack = jax.device_put(volume_stack, device=self.config.device)

        self.volume0_stack = jax.device_put(volume_stack, device=self.config.device)

        if L_stack is None:
            self.L_stack = jnp.zeros(
                (self.config.num_points, 3, 3), device=self.config.device
            )
        else:
            self.L_stack = jax.device_put(L_stack, device=self.config.device)

        if stress_stack is None:
            self.stress_stack = jnp.zeros(
                (self.config.num_points, 3, 3), device=self.config.device
            )
        else:
            self.stress_stack = jax.device_put(stress_stack, device=self.config.device)

        if F_stack is None:
            F_stack = jnp.stack([jnp.eye(3)] * self.config.num_points)
            self.F_stack = jax.device_put(F_stack, device=self.config.device)
        else:
            self.F_stack = jax.device_put(F_stack, device=self.config.device)

    def refresh(self) -> Self:
        """Zero velocity gradient"""
        return eqx.tree_at(
            lambda state: (state.L_stack),
            self,
            (self.L_stack.at[:].set(0.0)),
        )

    def calculate_volume(self: Self):
        """Calculate volume of particle given particles per cell and the cell size.
        Assumes all particles in a cell contribute equally to the volume."""
        volume_stack = (
            jnp.ones(self.config.num_points)
            * (self.config.cell_size**self.config.dim)
            / self.config.ppc
        )
        return volume_stack

    def get_solid_volume_fraction_stack(self: Self, rho_p: jnp.float32) -> chex.Array:
        """Get solid volume fraction of the particles.

        Args:
            rho_p (jnp.float32): particle density

        Returns:
            chex.Array: Solid volume fraction
        """
        density_stack = self.mass_stack / self.volume_stack
        return density_stack / rho_p
