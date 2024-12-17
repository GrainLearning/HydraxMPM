import dataclasses

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Generic, Optional, Self, Union, Unpack

from hydraxmpm.utils.math_helpers import get_pressure_stack

from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3PStack,
    TypeFloatScalarPStack,
    TypeFloatVectorPStack,
)
from ..config.mpm_config import MPMConfig


class Particles(eqx.Module):
    """Represents a collection of material points (particles) in an MPM simulation.

    These Lagrangian markers move across the background grid (see Solver class #TODO insert link).


    This class is typically created internally by the solver but can be instantiated directly if needed.

    Example usage:

    ```python
    import hydraxmpm as hdx
    import jax.numpy as jnp

    position_stack = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Example positions

    particles = hdx.Particles(
        position_stack=position_stack,
        # ... other particle data
    )
    ```

    Particle data is stored as attributes of this dataclass. Example of accessing stress and calculating pressure:

    ```python
    import hydraxmpm as hdx

    stress_stack = particles.stress_stack
    pressure_stack = hdx.get_pressure_stack(stress_stack)
    ```

    Attributes:
        config (:class:`MPMConfig`): The MPM configuration object.
        position_stack (jnp.ndarray): Spatial coordinate vectors (shape: `(num_particles, dim)`).
        velocity_stack (jnp.ndarray): Spatial velocity vectors (shape: `(num_particles, dim)`).
        force_stack (jnp.ndarray): External force vectors (shape: `(num_particles, dim)`).
        mass_stack (jnp.ndarray): Particle masses (shape: `(num_particles,)`).
        volume_stack (jnp.ndarray): Current particle volumes (shape: `(num_particles,)`).
        volume0_stack (jnp.ndarray): Initial particle volumes (shape: `(num_particles,)`).
        L_stack (jnp.ndarray): Velocity gradient tensors (shape: `(num_particles, dim, dim)`).
        stress_stack (jnp.ndarray): Cauchy stress tensors (shape: `(num_particles, dim, dim)`).
        F_stack (jnp.ndarray): Deformation gradient tensors (shape: `(num_particles, dim, dim)`).
        rho_p (float): Particle density (assuming constant particle density for all particles).
    """

    config: MPMConfig = eqx.field(static=True)

    position_stack: TypeFloatVectorPStack
    velocity_stack: TypeFloatVectorPStack
    force_stack: TypeFloatVectorPStack

    mass_stack: TypeFloatScalarPStack
    volume_stack: TypeFloatScalarPStack
    volume0_stack: TypeFloatScalarPStack

    L_stack: TypeFloatMatrix3x3PStack
    stress_stack: TypeFloatMatrix3x3PStack
    F_stack: TypeFloatMatrix3x3PStack

    rho_p: float = eqx.field(default=1, static=True, converter=lambda x: float(x))

    def __init__(
        self: Self,
        config: MPMConfig,
        position_stack: Optional[TypeFloatVectorPStack] = None,
        velocity_stack: Optional[TypeFloatVectorPStack] = None,
        force_stack: Optional[TypeFloatVectorPStack] = None,
        mass_stack: Optional[TypeFloatScalarPStack] = None,
        volume_stack: Optional[TypeFloatScalarPStack] = None,
        volume0_stack: Optional[TypeFloatScalarPStack] = None,
        stress_stack: Optional[TypeFloatMatrix3x3PStack] = None,
        L_stack: Optional[TypeFloatMatrix3x3PStack] = None,
        F_stack: Optional[TypeFloatMatrix3x3PStack] = None,
        pressure_ref: Optional[Union[TypeFloatScalarPStack, TypeFloat, float]] = None,
        density_ref: Optional[Union[TypeFloatScalarPStack, TypeFloat, float]] = None,
    ) -> Self:
        self.config = config

        if position_stack is None:
            position_stack = jnp.zeros((self.config.num_points, self.config.dim))

        self.position_stack = jax.device_put(position_stack, device=self.config.device)

        if velocity_stack is None:
            velocity_stack = jnp.zeros((self.config.num_points, self.config.dim))

        self.velocity_stack = jax.device_put(velocity_stack, device=self.config.device)

        if force_stack is None:
            force_stack = jnp.zeros((self.config.num_points, self.config.dim))
        self.force_stack = jax.device_put(force_stack, device=self.config.device)

        # volumes get discretized as grid by default
        if volume_stack is None:
            volume_stack = (
                jnp.ones(self.config.num_points)
                * (self.config.cell_size**self.config.dim)
                / self.config.ppc
            )
        else:
            volume_stack = jax.device_put(volume_stack, device=self.config.device)

        self.volume_stack = jax.device_put(volume_stack, device=self.config.device)

        if volume0_stack is None:
            volume0_stack = volume_stack

        self.volume0_stack = volume0_stack

        if density_ref is not None:
            mass_stack = density_ref * volume_stack
        if mass_stack is None:
            mass_stack = jnp.zeros(self.config.num_points)

        self.mass_stack = jax.device_put(mass_stack, device=self.config.device)

        if L_stack is None:
            L_stack = jnp.zeros((self.config.num_points, 3, 3))

        self.L_stack = jax.device_put(L_stack, device=self.config.device)

        # initialize form from a reference pressure (or stack)
        # reference stress
        # or zeros
        if pressure_ref is not None:
            if eqx.is_array(pressure_ref):
                stress_stack = jax.vmap(lambda x: -x * jnp.eye(3))(pressure_ref)
            else:
                zeros_stack = jnp.zeros(
                    (self.config.num_points, 3, 3), device=self.config.device
                )
                stress_stack = jax.vmap(lambda x: -pressure_ref * jnp.eye(3))(
                    zeros_stack
                )

        elif stress_stack is None:
            stress_stack = jnp.zeros((self.config.num_points, 3, 3))

        self.stress_stack = jax.device_put(stress_stack, device=self.config.device)

        if F_stack is None:
            F_stack = jnp.stack([jnp.eye(3)] * self.config.num_points)

        self.F_stack = jax.device_put(F_stack, device=self.config.device)

    def refresh(self) -> Self:
        """Zero velocity gradient"""
        return eqx.tree_at(
            lambda state: (state.L_stack, state.force_stack),
            self,
            (self.L_stack.at[:].set(0.0), self.force_stack.at[:].set(0.0)),
        )

    def get(self, key: str, index: slice = None):
        """Get specific quantity from particles

        Following particle attributes are supported:
        `position_stack`, `velocity_stack`,`force_stack`, `mass_stack`
        `volume_stack`, `volume0_stack`, `L_stack` (velocity gradient),
        `stress_stack` (cauchy stress), `F_stack` (deformation gradient)

        Following derived attributes are supported:
        `phi_stack` (solid volume fraction), if rho_p is defined, otherwise returns density
        `density_stack`,`pressure_stack`


        Args:
            key: key to return
            index: specific index in array to return. Defaults to None.
        """
        if index is None:
            index = slice(None)
        if key == "density_stack":
            X_stack = self.mass_stack / self.volume_stack
        if key == "phi_stack":
            X_stack = self.get_solid_volume_fraction_stack()
        elif key == "pressure_stack":
            X_stack = get_pressure_stack(self.stress_stack)
        else:
            X_stack = self.__getattribute__(key)
        return X_stack.at[index].get()

    def get_solid_volume_fraction_stack(self: Self) -> TypeFloatScalarPStack:
        """Get solid volume fraction

        $$
            \\phi = m/v
        $$

        where m is the particle mass, and v is the current volume.

        """
        density_stack = self.mass_stack / self.volume_stack
        return density_stack / self.rho_p

    def replace(self, **kwargs: Unpack):
        """Replace values in dataframe"""
        return dataclasses.replace(self, **kwargs)
