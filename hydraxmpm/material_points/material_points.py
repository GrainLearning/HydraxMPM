# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:

    This module includes state definitions for material points in MPM simulations.

    The `BaseMaterialPointState` class, defining basic material point properties like position, mass, and velocity.

    The `MaterialPointState` class extends the base class to include additional attributes
    such as force, volume, stress, deformation gradient, and velocity gradient.

    The `RigidMaterialPointState` class defines state for rigid body material points. This hos no deformation or stress
    and should not be included in the particle-to-grid/grid-to-particle transfers or constitutive model update.


    References:
    - De Vaucorbeil, Alban, et al. "Material point method after 25 years: Theory, implementation, and applications."

"""

import equinox as eqx
import jax

from typing import Self, Optional

from jaxtyping import Array, Float

import jax.numpy as jnp

import warnings

from ..utils.math_helpers import (
    get_pressure_stack,
    get_q_vm_stack,
    get_hencky_strain_stack,
    get_volumetric_strain_stack,
    get_strain_rate_tensor_stack,
    get_shear_strain_vm_stack,
)


class BaseMaterialPointState(eqx.Module):
    """Base state for material point state in MPM simulations.

    Attributes:
        position_stack: Position coordinates of material points.
        mass_stack: Mass of each material point.
        velocity_stack: Velocity vectors of material points.
    """

    position_stack: Float[Array, "num_points dim"]
    mass_stack: Float[Array, "num_points"]
    velocity_stack: Optional[Float[Array, "num_points dim"]]

    @property
    def num_points(self) -> int:
        """Gives total number of material points"""
        return self.position_stack.shape[0]

    @property
    def dim(self) -> int:
        """Gives dimension of the material points"""
        return self.position_stack.shape[1]


class RigidMaterialPointState(BaseMaterialPointState):
    """
    Rigid body state for material points in MPM simulations.

    Inherits from the BaseMaterialPointState.

    Note if no normal_stack is provided, the grid contact algorim uses mass gradients
    to approximate normals for contact handling (See `GridContact`)

    Attributes:
        normal_stack: optional normal vectors for rigid body surface points.
    """

    normal_stack: Optional[Float[Array, "num_points dim"]] = None

    @classmethod
    def create(
        cls,
        *,
        position_stack: Optional[Float[Array, "num_points dim"]] = None,
        normal_stack: Optional[Float[Array, "num_points dim"]] = None,
    ) -> Self:
        """Helper function to create RigidMaterialPointState with default values."""
        if position_stack is None:
            position_stack = jnp.array([[0.0, 0.0, 0.0]])
            warning = (
                "No position_stack provided for RigidMaterialPoints. "
                "Defaulting to a single particle at the origin."
            )
            warnings.warn(warning)

        position_stack = jnp.array(position_stack)

        num_points, dim = position_stack.shape

        if normal_stack is None:
            normal_stack = jnp.zeros((num_points, dim))

        return cls(
            position_stack=position_stack,
            mass_stack=jnp.ones((num_points,)),
            velocity_stack=jnp.zeros((num_points, dim)),
            normal_stack=normal_stack,
        )


class MaterialPointState(BaseMaterialPointState):
    """

    Standard material point state for MPM simulations.

    Inherits from the BaseMaterialPointState.

    - Volume initialization defaults to uniform distribution based on cell size and points per cell.
    - This is calculated from input parameters `cell_size` and `points_per_cell` if provided.
    - Mass initiation defaults to density if `mass_stack` is not provided.

    Attributes:

         force_stack: External force vectors.
         mass_stack: material point masses, assumed to remain constant throughout the simulation.
         volume_stack: Current particle volumes.
         volume0_stack: Initial particle volumes.
         L_stack: Velocity gradient tensors.
         stress_stack: Cauchy stress tensors.
         F_stack: Deformation gradient tensors.
         density_per_particle: Particle density. Defaults to 1000.0 if not provided.
         kwargs:
            - cell_size: Size of the grid cell. Defaults to 1.0 if not provided.
            - points_per_cell: Number of material points per cell. Defaults to 4 if not provided.
            - density_stack: Density per particle. Defaults to 1000.0 if not provided.
    """

    force_stack: Float[Array, "num_points dim"]
    volume_stack: Float[Array, "num_points"]
    volume0_stack: Float[Array, "num_points"]
    L_stack: Float[Array, "num_points 3 3"]
    stress_stack: Float[Array, "num_points 3 3"]
    F_stack: Float[Array, "num_points 3 3"]

    @classmethod
    def create(
        cls,
        *,
        position_stack: Optional[Float[Array, "num_points dim"]] = None,
        velocity_stack: Optional[Float[Array, "num_points dim"]] = None,
        force_stack: Optional[Float[Array, "num_points dim"]] = None,
        mass_stack: Optional[Float[Array, "num_points"]] = None,
        volume_stack: Optional[Float[Array, "num_points"]] = None,
        volume0_stack: Optional[Float[Array, "num_points"]] = None,
        L_stack: Optional[Float[Array, "num_points 3 3"]] = None,
        stress_stack: Optional[Float[Array, "num_points 3 3"]] = None,
        F_stack: Optional[Float[Array, "num_points 3 3"]] = None,
        **kwargs,
    ) -> Self:
        """Helper function to create MaterialPointState with default values."""

        if position_stack is None:
            position_stack = jnp.array([[0.0, 0.0, 0.0]])

        position_stack = jnp.array(position_stack)

        num_points, dim = position_stack.shape

        # Initialize with empty/default arrays if not provided
        velocity_stack = (
            velocity_stack
            if velocity_stack is not None
            else jnp.zeros((num_points, dim))
        )

        force_stack = (
            force_stack if force_stack is not None else jnp.zeros((num_points, dim))
        )

        stress_stack = (
            stress_stack if stress_stack is not None else jnp.zeros((num_points, 3, 3))
        )

        L_stack = L_stack if L_stack is not None else jnp.zeros((num_points, 3, 3))

        F_stack = (
            F_stack if F_stack is not None else jnp.tile(jnp.eye(3), (num_points, 1, 1))
        )

        # Default volume calculation if not provided.
        # Assumes uniform distribution based on cell size and points per cell.
        if volume_stack is None:
            cell_size = kwargs.get("cell_size", 1.0)
            points_per_cell = kwargs.get("ppc", 4.0)
            default_volume = (cell_size**2) / points_per_cell
            volume_stack = jnp.ones(num_points) * default_volume

        volume0_stack = volume0_stack if volume0_stack is not None else volume_stack

        # Default mass calculation if not provided, using density
        if mass_stack is None:
            density_stack = kwargs.get("density_stack", jnp.full((num_points,), 1000.0))
            mass_stack = volume_stack * density_stack

        return cls(
            position_stack=position_stack,
            velocity_stack=velocity_stack,
            force_stack=force_stack,
            mass_stack=mass_stack,
            volume_stack=volume_stack,
            volume0_stack=volume0_stack,
            L_stack=L_stack,
            stress_stack=stress_stack,
            F_stack=F_stack,
        )

    @property
    def density_stack(self):
        """Get current density of material points."""
        return self.mass_stack / (self.volume_stack + 1e-16)

    @property
    def density_ref_stack(self):
        """Get initial density of material points."""
        return self.mass_stack / (self.volume0_stack + 1e-16)

    @property
    def pressure_stack(self):
        """Pressure of material points. Compression positive.
        (see `get_pressure_stack` in utils/math_helpers.py)
        """
        return get_pressure_stack(self.stress_stack)

    @property
    def KE_density_stack(self):
        """Kinetic energy density of material points."""
        NotImplementedError("KE_density_stack property not implemented yet.")
        pass
        # return get_KE_stack(self.rho_stack, self.velocity_stack)

    @property
    def KE_stack(self):
        """Kinetic energy of material points."""
        NotImplementedError("KE_stack property not implemented yet.")
        pass
        # return get_KE_stack(self.mass_stack, self.velocity_stack)

    @property
    def q_stack(self):
        """Triaxial shear stress invariant of material points."""
        return get_q_vm_stack(self.stress_stack)
        # NotImplementedError("q_stack property not implemented yet.")
        # pass

    @property
    def q_p_stack(self):
        """Shear stress to pressure ratio of material points."""
        NotImplementedError("q_p_stack property not implemented yet.")
        pass

        # return self.q_stack / self.p_stack

    @property
    def eps_stack(self):
        """Hencky strain tensor of material points."""
        # NotImplementedError("eps_stack property not implemented yet.")
        return get_hencky_strain_stack(self.F_stack)

    @property
    def eps_v_stack(self):
        """Volumetric strain of material points."""

        return get_volumetric_strain_stack(self.eps_stack)

    @property
    def deps_dt_stack(self):
        """Strain rate tensor of material points."""
        return get_strain_rate_tensor_stack(self.L_stack)

    @property
    def shear_strain_rate_stack(self):
        return get_shear_strain_vm_stack(self.deps_dt_stack)
    @property
    def shear_strain_stack(self):
        return get_shear_strain_vm_stack(self.eps_stack)
    # @property
    # def gamma_stack(self):
    #     NotImplementedError("gamma_stack property not implemented yet.")
    #     pass
    #     # return get_scalar_shear_strain_stack(self.eps_stack)

    # @property
    # def dgamma_dt_stack(self):
    #     NotImplementedError("dgamma_dt_stack property not implemented yet.")
    #     pass
    #     # return get_scalar_shear_strain_stack(self.deps_dt_stack)

    # @property
    # def viscosity_stack(self):
    #     NotImplementedError("viscosity_stack property not implemented yet.")
    #     pass
    #     # return (jnp.sqrt(3) * self.q_stack) / self.dgamma_dt_stack
