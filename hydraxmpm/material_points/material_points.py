# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Self

import equinox as eqx
import jax
import jax.numpy as jnp

from ..common.base import Base
from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3PStack,
    TypeFloatScalarPStack,
    TypeFloatVector,
    TypeFloatVectorPStack,
    TypeInt,
)
from ..utils.math_helpers import (
    get_hencky_strain_stack,
    get_inertial_number_stack,
    get_KE_stack,
    get_pressure_stack,
    get_q_vm_stack,
    get_scalar_shear_strain_stack,
    get_strain_rate_from_L_stack,
    get_volumetric_strain_stack,
    get_double_contraction_stack,
    get_double_contraction,
)


class MaterialPoints(Base):
    """Collection of material points (particles) in an MPM simulation.

    Attributes:
        position_stack: Spatial coordinate vectors (shape: `(num_points, dim)`).
        velocity_stack: Spatial velocity vectors (shape: `(num_points, dim)`).
        force_stack: External force vectors (shape: `(num_points, dim)`).
        mass_stack: material point masses (shape: `(num_points,)`).
        volume_stack: Current particle volumes (shape: `(num_points,)`).
        volume0_stack: Initial particle volumes (shape: `(num_points,)`).
        L_stack: Velocity gradient tensors (shape: `(num_points, dim, dim)`).
        stress_stack: Cauchy stress tensors (shape: `(num_points, dim, dim)`).
        F_stack: Deformation gradient tensors (shape: `(num_points, dim, dim)`).
        rho_p: Particle density (assuming constant particle density for all particles).
        rho_0: Reference pressure for particles (scalar or array of shape: `(num_points,)`).
        rho_0: Reference density for particles (scalar or array of shape: `(num_points,)`).
    """

    position_stack: TypeFloatVectorPStack
    velocity_stack: TypeFloatVectorPStack
    force_stack: TypeFloatVectorPStack

    mass_stack: TypeFloatScalarPStack
    volume_stack: TypeFloatScalarPStack
    volume0_stack: TypeFloatScalarPStack

    L_stack: TypeFloatMatrix3x3PStack
    stress_stack: TypeFloatMatrix3x3PStack
    F_stack: TypeFloatMatrix3x3PStack

    dim: TypeInt = eqx.field(init=False, static=True, default=3)
    num_points: TypeInt = eqx.field(init=False, static=True, default=1)

    ref_gravity_pe_pos: Optional[TypeFloatVector] = None
    ref_gravity: Optional[TypeFloatVector] = None

    def __init__(
        self: Self,
        position_stack: Optional[TypeFloatVectorPStack] = None,
        velocity_stack: Optional[TypeFloatVectorPStack] = None,
        force_stack: Optional[TypeFloatVectorPStack] = None,
        mass_stack: Optional[TypeFloatScalarPStack] = None,
        stress_stack: Optional[TypeFloatMatrix3x3PStack] = None,
        # Use kwargs for less common parameters
        **kwargs,
    ) -> Self:
        # Initialize required fields

        if position_stack is None:
            position_stack = jnp.array([[0.0, 0.0, 0.0]])

        self.position_stack = position_stack
        self.num_points, self.dim = position_stack.shape

        # Set defaults for explicit args
        self.velocity_stack = (
            velocity_stack
            if velocity_stack is not None
            else jnp.zeros((self.num_points, self.dim))
        )

        self.force_stack = (
            force_stack
            if force_stack is not None
            else jnp.zeros((self.num_points, self.dim))
        )

        self.mass_stack = (
            mass_stack if mass_stack is not None else jnp.ones(self.num_points)
        )
        p_stack = kwargs.get("p_stack", None)
        if p_stack is not None:
            if eqx.is_array(p_stack):
                stress_stack = jax.vmap(lambda x: -x * jnp.eye(3))(p_stack)
            else:
                stress_stack = jnp.tile(jnp.eye(3), (self.num_points, 1, 1)) * -p_stack

        self.stress_stack = (
            stress_stack
            if stress_stack is not None
            else jnp.zeros((self.num_points, 3, 3))
        )

        self.volume_stack = kwargs.get("volume_stack", jnp.ones(self.num_points))
        self.volume0_stack = kwargs.get("volume0_stack", self.volume_stack.copy())
        self.L_stack = kwargs.get("L_stack", jnp.zeros((self.num_points, 3, 3)))

        self.F_stack = kwargs.get(
            "F_stack", jnp.tile(jnp.eye(3), (self.num_points, 1, 1))
        )

        self.ref_gravity_pe_pos = kwargs.get("ref_gravity_pe_pos")
        self.ref_gravity = kwargs.get("ref_gravity")
        super().__init__(**kwargs)

    def init_stress_from_p_0(self: Self, p_0) -> Self:
        # initialize stress tensor from hydrostatic pressure
        # assumes no deviatoric stresses
        p_0_array = jnp.array([p_0]).flatten()
        if p_0_array.shape[0] == self.num_points:
            new_stress_stack = jax.vmap(lambda x: -x * jnp.eye(3))(p_0_array)
        else:
            new_stress_stack = jax.vmap(lambda x: -p_0_array * jnp.eye(3))(
                self.stress_stack
            )

        return eqx.tree_at(
            lambda state: state.stress_stack,
            self,
            (new_stress_stack),
        )

    def init_volume_from_cellsize(
        self, cell_size: TypeFloat, ppc: int, init_volume0=True
    ) -> Self:
        """
        Discretizes the particles into cells so volumes are distributed evenly
        for a given number of particles per cell
        """

        new_volume_stack = (jnp.ones(self.num_points) * cell_size**self.dim) / ppc
        if init_volume0:
            new_volume0_stack = new_volume_stack

        return eqx.tree_at(
            lambda state: (state.volume_stack, state.volume0_stack),
            self,
            (new_volume_stack, new_volume0_stack),
        )

    def init_mass_from_rho_0(self: Self, rho_0) -> Self:
        """
        Discretizes the particles into cells so densities are distributed evenly
        for a given number of particles per cell
        """
        new_mass_stack = rho_0 * self.volume_stack
        return eqx.tree_at(lambda state: (state.mass_stack), self, (new_mass_stack))

    def _refresh(self) -> Self:
        """Zero velocity gradient"""
        return eqx.tree_at(
            lambda state: (state.L_stack, state.force_stack),
            self,
            (self.L_stack.at[:].set(0.0), self.force_stack.at[:].set(0.0)),
        )

    def update_L_and_F_stack(self, L_stack_next, dt):
        # TODO a better name for this function?
        def update_F_volume(L_next, F_prev, volume0):
            F_next = (jnp.eye(3) + L_next * dt) @ F_prev
            volume_next = jnp.linalg.det(F_next) * volume0
            return F_next, volume_next

        F_stack_next, volume_stack_next = jax.vmap(update_F_volume)(
            L_stack_next, self.F_stack, self.volume0_stack
        )

        return self.replace(
            L_stack=L_stack_next, F_stack=F_stack_next, volume_stack=volume_stack_next
        )

    @property
    def rho_stack(self):
        return self.mass_stack / self.volume_stack

    @property
    def rho0_stack(self):
        return self.mass_stack / self.volume0_stack

    @property
    def p_stack(self):
        return get_pressure_stack(self.stress_stack)

    @property
    def KE_density_stack(self):
        return get_KE_stack(self.rho_stack, self.velocity_stack)

    @property
    def KE_stack(self):
        return get_KE_stack(self.mass_stack, self.velocity_stack)

    @property
    def q_stack(self):
        return get_q_vm_stack(self.stress_stack)

    @property
    def q_p_stack(self):
        return self.q_stack / self.p_stack

    @property
    def eps_stack(self):
        return get_hencky_strain_stack(self.F_stack)[0]

    @property
    def eps_v_stack(self):
        return get_volumetric_strain_stack(self.eps_stack)

    @property
    def deps_dt_stack(self):
        return get_strain_rate_from_L_stack(self.L_stack)

    @property
    def gamma_stack(self):
        return get_scalar_shear_strain_stack(self.eps_stack)

    @property
    def dgamma_dt_stack(self):
        return get_scalar_shear_strain_stack(self.deps_dt_stack)

    @property
    def viscosity_stack(self):
        return (jnp.sqrt(3) * self.q_stack) / self.dgamma_dt_stack

    # def work_ext_stack(self, dt, **kwargs):
    #     # external work  (e.g., potential energy due to gravity)
    #     def vmap_W_ext(force, vel):
    #         # relative displacement
    #         rel_disp = vel * dt
    #         return jnp.dot(force, rel_disp)

    #     return jax.vmap(vmap_W_ext)(self.force_stack, self.velocity_stack)

    def PE_ext_stack(self, dt, **kwargs):
        return self.work_ext_stack(dt)

    @property
    def PE_grav_stack(self):
        if self.ref_gravity is None:
            return

        def vmap_grav_pe(pos, mass):
            f_grav = mass * self.ref_gravity  # neg
            displacement = self.ref_gravity_pe_pos - pos
            # return jnp.dot(f_grav, displacement)  # positive if in same direction

            return jnp.dot(f_grav, displacement)

        return jax.vmap(vmap_grav_pe)(self.position_stack, self.mass_stack)

    def PE_stack(self, dt, W_stack, **kwargs):
        assert W_stack is not None, (
            "Please set approx_strain_energy_density=True for the material points"
        )
        return W_stack * self.volume_stack + self.PE_grav_stack

    def KE_PE_Stack(self, dt, W_stack, **kwargs):
        return self.KE_stack / self.PE_stack(dt, W_stack)

    def deps_stack(self, dt, **kwargs):
        return self.deps_dt_stack * dt

    def phi_stack(self, rho_p, **kwargs):
        return self.rho_stack / rho_p

    def specific_volume_stack(self, rho_p, **kwargs):
        return 1.0 / self.phi_stack(rho_p)

    def inertial_number_stack(self, d, rho_p, **kwargs):
        return get_inertial_number_stack(
            self.p_stack,
            self.dgamma_dt_stack,
            d,
            rho_p,
        )

    def deps_p_dt_stack(self, dt, eps_e_stack=None, eps_e_stack_prev=None, **kwargs):
        """Plastic strain rate tensor"""

        if eps_e_stack is None:
            eps_e_stack = jnp.zeros((3, 3))

        if eps_e_stack_prev is None:
            eps_e_stack_prev = jnp.zeros((3, 3))

        return (eps_e_stack - eps_e_stack_prev) * dt

    def dgamma_p_dt_stack(self, dt, eps_e_stack=None, eps_e_stack_prev=None, **kwargs):
        """Plastic scalar shear strain rate"""

        return get_scalar_shear_strain_stack(
            self.deps_p_dt_stack(dt, eps_e_stack, eps_e_stack_prev, **kwargs)
        )

    def deps_p_v_dt_stack(self, dt, eps_e_stack, eps_e_stack_prev=None, **kwargs):
        """Plastic scalar volumetric strain rate"""

        return get_volumetric_strain_stack(
            self.deps_p_dt_stack(dt, eps_e_stack, eps_e_stack_prev, **kwargs)
        )
