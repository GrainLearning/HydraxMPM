# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Optional, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3,
    TypeFloatMatrix3x3PStack,
    TypeFloatScalarNStack,
    TypeInt,
)
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..forces.force import Force
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from .mpm_solver import MPMSolver


class USL_ASFLIP(MPMSolver):
    """Explicit Update Stress Last (USL) Affine."""

    alpha: float = eqx.field(static=True)
    phi_c: float = eqx.field(static=True)
    beta_min: float = eqx.field(static=True)
    beta_max: float = eqx.field(static=True)

    Dp: TypeFloatMatrix3x3
    Dp_inv: TypeFloatMatrix3x3
    Bp_stack: TypeFloatMatrix3x3PStack
    
    J_stack: TypeFloatScalarNStack

    def __init__(
        self,
        *,
        dim,
        material_points: MaterialPoints,
        grid: Grid,
        constitutive_laws: Optional[
            Tuple[ConstitutiveLaw, ...] | ConstitutiveLaw
        ] = None,
        forces: Optional[Tuple[Force, ...]] = None,
        ppc=1,
        shapefunction="cubic",
        output_dict: Optional[dict | Tuple[str, ...]] = None,
        alpha: Optional[TypeFloat] = 1.0,
        phi_c: TypeFloat = 0.5,
        beta_min: TypeFloat = 0.0,
        beta_max: TypeFloat = 0.0,
        **kwargs,
    ):
        super().__init__(
            material_points=material_points,
            grid=grid,
            constitutive_laws=constitutive_laws,
            forces=forces,
            dim=dim,
            ppc=ppc,
            shapefunction=shapefunction,
            output_dict=output_dict,
            **kwargs,
        )

        self.alpha = alpha
        self.phi_c = phi_c
        self.beta_min = beta_min
        self.beta_max = beta_max

        if self.shapefunction not in ["cubic", "quadratic"]:
            raise NotImplementedError("Only cubic shapefunctions supported with ASFLIP")

        Dp = kwargs.get("Dp", None)
        Dp_inv = kwargs.get("Dp_inv", None)
        Bp_stack = kwargs.get("Bp_stack", None)

        if Dp is None:
            if self.shapefunction == "cubic":
                self.Dp = (
                    (1.0 / 3.0) * self.grid.cell_size * self.grid.cell_size * jnp.eye(3)
                )
            elif self.shapefunction == "quadratic":
                self.Dp = (
                    (1.0 / 4.0) * self.grid.cell_size * self.grid.cell_size * jnp.eye(3)
                )

        else:
            self.Dp = Dp

        if Dp_inv is None:
            self.Dp_inv = jnp.linalg.inv(self.Dp)
        else:
            self.Dp_inv = Dp_inv

        if Bp_stack is None:
            self.Bp_stack = jnp.zeros((self.material_points.num_points, 3, 3))
        else:
            self.Bp_stack = Bp_stack
        

        self.J_stack = jnp.zeros(self.grid.num_cells)

    @jax.checkpoint
    def update(self: Self, step: TypeInt = 0, dt: TypeFloat = 1e-3) -> Self:
        material_points = self.material_points._refresh()

        material_points, forces = self._update_forces_on_points(
            material_points=material_points,
            grid=self.grid,
            forces=self.forces,
            step=step,
            dt=dt,
        )

        self, new_shape_map, grid = self.p2g(
            material_points=material_points, grid=self.grid, dt=dt
        )

        grid, forces = self._update_forces_grid(
            material_points=material_points, grid=grid, forces=forces, step=step, dt=dt
        )

        self, material_points = self.g2p(
            material_points=material_points, grid=grid, shape_map=new_shape_map, dt=dt
        )

        material_points, constitutive_laws = self._update_constitutive_laws(
            material_points, self.constitutive_laws, dt=dt
        )

        return eqx.tree_at(
            lambda state: (
                state.material_points,
                state.grid,
                state.constitutive_laws,
                state.forces,
                state.shape_map,
            ),
            self,
            (material_points, grid, constitutive_laws, forces, new_shape_map),
        )

    def p2g(self, material_points, grid, dt):
        def vmap_intr_p2g(point_id, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = material_points.mass_stack.at[point_id].get()
            intr_volumes = material_points.volume_stack.at[point_id].get()
            intr_velocities = material_points.velocity_stack.at[point_id].get()
            intr_ext_forces = material_points.force_stack.at[point_id].get()
            intr_stresses = material_points.stress_stack.at[point_id].get()
            intr_F_stack = material_points.F_stack.at[point_id].get()

            intr_Bp = self.Bp_stack.at[point_id].get()  # APIC affine matrix

            affine_velocity = (
                intr_Bp @ jnp.linalg.inv(self.Dp)
            ) @ intr_dist  # intr_dist is 3D

            if self.dim == 2:
                affine_velocity = eqx.error_if(
                    affine_velocity,
                    jnp.abs(affine_velocity.at[2].get()) > 1e-12,
                    "error out of plane affine velocity detected for plane strain",
                )
            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (
                intr_velocities + affine_velocity.at[: self.dim].get()
            )

            scaled_ext_force = intr_shapef * intr_ext_forces

            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            if self.dim == 2:
                scaled_int_force = eqx.error_if(
                    scaled_int_force,
                    jnp.abs(scaled_int_force.at[2].get()) > 1e-12,
                    "error out of plane forces detected for plane strain",
                )

            scaled_total_force = (
                scaled_int_force.at[: self.dim].get() + scaled_ext_force
            )

            scaled_normal = (intr_shapef_grad * intr_masses).at[: self.dim].get()

            scaled_Jm = scaled_mass * jnp.linalg.det(intr_F_stack)

            return scaled_mass, scaled_moments, scaled_total_force, scaled_normal, scaled_Jm

        def sum_interactions(stack, scaled_stack):
            return (
                jnp.zeros_like(stack)
                .at[new_shape_map._intr_hash_stack]
                .add(scaled_stack)
            )
        # note the interactions and shapefunctions are calculated on the
        # p2g to reduce computational overhead.
        (new_shape_map,p2g_out_stacks)= self.shape_map.vmap_interactions_and_scatter(
            vmap_intr_p2g, material_points, grid
        )

        (
            scaled_mass_stack,
            scaled_moment_stack,
            scaled_total_force_stack,
            scaled_normal_stack,
            scaled_Jm_stack,
        ) = p2g_out_stacks
  



        # sum
        new_mass_stack = sum_interactions(grid.mass_stack, scaled_mass_stack)
        new_moment_stack = sum_interactions(grid.moment_stack, scaled_moment_stack)
        new_force_stack = sum_interactions(
            self.grid.moment_stack, scaled_total_force_stack
        )
        new_normal_stack = sum_interactions(grid.normal_stack, scaled_normal_stack)
        # new_node_Jm_stack = sum_interactions(self.J_stack, scaled_Jm_stack)

        # def divide(X_generic, mass):
   
        #     result = X_generic / (mass + 1e-22)
        #     return result
            
        # new_node_J_stack = jax.vmap(divide)(new_node_Jm_stack, new_mass_stack)
        nodes_moment_nt_stack = new_moment_stack + new_force_stack * dt

        # new_solver = eqx.tree_at(
        #     lambda state: (state.J_stack),
        #     self,
        #     (new_node_J_stack),
        # )

        return self,new_shape_map, eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
                state.normal_stack,
            ),
            grid,
            (new_mass_stack, new_moment_stack, nodes_moment_nt_stack, new_normal_stack),
        )

    def g2p(self, material_points, grid, shape_map, dt) -> Tuple[Self, MaterialPoints]:
        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = grid.mass_stack.at[intr_hashes].get()
            intr_moments = grid.moment_stack.at[intr_hashes].get()
            intr_moments_nt = grid.moment_nt_stack.at[intr_hashes].get()
            intr_J = self.J_stack.at[intr_hashes].get()

            # intr_vels = intr_moments / (intr_masses + grid.small_mass_cutoff)
            intr_vels = jax.lax.cond(
                intr_masses > grid.small_mass_cutoff,
                lambda x: x / (intr_masses + 1e-22),
                lambda x: jnp.zeros_like(x),
                intr_moments,
            )
            # intr_vels_nt = intr_moments_nt / (intr_masses + grid.small_mass_cutoff)
            intr_vels_nt = jax.lax.cond(
                intr_masses > grid.small_mass_cutoff,
                lambda x: x / (intr_masses + 1e-22),
                lambda x: jnp.zeros_like(x),
                intr_moments_nt,
            )

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            intr_delta_vels = intr_vels_nt - intr_vels
            intr_scaled_delta_vels = intr_shapef * intr_delta_vels
            
            intr_scaled_J  = intr_shapef * intr_J

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                self._padding,
                mode="constant",
                constant_values=0,
            )

            # APIC affine matrix
            intr_Bp = (
                intr_shapef
                * intr_vels_nt_padded.reshape(-1, 1)
                @ intr_dist.reshape(-1, 1).T
            )

            intr_scaled_velgrad = intr_Bp @ self.Dp_inv

            return (
                intr_scaled_delta_vels,
                intr_scaled_vels_nt,
                intr_scaled_velgrad,
                intr_Bp,
                intr_scaled_J
            )

        (
            new_intr_scaled_delta_vel_stack,
            new_intr_scaled_vel_nt_stack,
            new_intr_scaled_velgrad_stack,
            new_intr_Bp_stack,
            new_intr_scaled_J_stack
        ) = shape_map.vmap_intr_gather(vmap_intr_g2p)
        if self.error_check:
            new_intr_scaled_delta_vel_stack = eqx.error_if(
                new_intr_scaled_delta_vel_stack,
                ~jnp.isfinite(new_intr_scaled_delta_vel_stack),
                "new_intr_scaled_delta_vel_stack is non finite",
            )
            new_intr_scaled_vel_nt_stack = eqx.error_if(
                new_intr_scaled_vel_nt_stack,
                ~jnp.isfinite(new_intr_scaled_vel_nt_stack),
                "new_intr_scaled_vel_nt_stack is non finite",
            )
            new_intr_scaled_velgrad_stack = eqx.error_if(
                new_intr_scaled_velgrad_stack,
                ~jnp.isfinite(new_intr_scaled_velgrad_stack),
                "new_intr_scaled_velgrad_stack is non finite",
            )
            new_intr_Bp_stack = eqx.error_if(
                new_intr_Bp_stack,
                ~jnp.isfinite(new_intr_Bp_stack),
                "new_intr_Bp_stack is non finite",
            )

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0,0))
        def vmap_particles_update(
            intr_delta_vels_reshaped,
            intr_vels_nt_reshaped,
            intr_velgrad_reshaped,
            intr_Bp,
            new_intr_scaled_J,
            p_velocities,
            p_positions,
            p_F,
            p_volumes_orig,
        ):
            # Update particle quantities
            p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

            delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
            vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

            p_Bp_next = jnp.sum(intr_Bp, axis=0)
            p_J_bar_next = jnp.sum(new_intr_scaled_J, axis=0)

            if self.dim == 2:
                p_Bp_next = p_Bp_next.at[2, 2].set(0.0)

            if self.error_check:
                p_velocities = eqx.error_if(
                    p_velocities,
                    ~jnp.isfinite(p_velocities),
                    "p_velocities is non finite",
                )

            T = self.alpha * (p_velocities + delta_vels - vels_nt)
            p_velocities_next = vels_nt + T

            if self.error_check:
                p_velgrads_next = eqx.error_if(
                    p_velgrads_next,
                    ~jnp.isfinite(p_velgrads_next),
                    "p_velgrads_next is non finite",
                )
            if self.dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0.0)

            # p_F_next = (jnp.eye(3) + p_velgrads_next * dt) @ p_F
            p_F_next = jax.scipy.linalg.expm(p_velgrads_next * dt) @ p_F
            
            # p_J_next = jnp.linalg.det(p_F_next)
            # p_F_next = ((p_J_bar_next/ p_J_next)**(1/3)) * p_F_next
            
            if self.dim == 2:
                p_F_next = p_F_next.at[2, 2].set(1)

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig

            if self.error_check:
                p_volumes_next = eqx.error_if(
                    p_volumes_next,
                    ~jnp.isfinite(p_volumes_next),
                    "p_volumes_next is non finite",
                )
            # solid volume fraction
            phi = p_volumes_orig / p_volumes_next  # mass is constant

            Beta_p = jax.lax.cond(
                phi < self.phi_c, lambda: self.beta_max, lambda: self.beta_min
            )

            vel_update = vels_nt + Beta_p * T

            p_positions_next = p_positions + vel_update * dt

            return (
                p_velocities_next,
                p_positions_next,
                p_F_next,
                p_volumes_next,
                p_velgrads_next,
                p_Bp_next,
            )

        (
            new_velocity_stack,
            new_position_stack,
            new_F_stack,
            new_volume_stack,
            new_L_stack,
            new_Bp_stack,
        ) = vmap_particles_update(
            new_intr_scaled_delta_vel_stack.reshape(
                -1, shape_map._window_size, self.dim
            ),
            new_intr_scaled_vel_nt_stack.reshape(-1, shape_map._window_size, self.dim),
            new_intr_scaled_velgrad_stack.reshape(-1, shape_map._window_size, 3, 3),
            new_intr_Bp_stack.reshape(-1, shape_map._window_size, 3, 3),
            new_intr_scaled_J_stack.reshape(-1, shape_map._window_size),
            material_points.velocity_stack,
            material_points.position_stack,
            material_points.F_stack,
            material_points.volume0_stack,
        )

        new_solver = eqx.tree_at(
            lambda state: (state.Bp_stack),
            self,
            (new_Bp_stack),
        )
        if self.error_check:
            new_volume_stack = eqx.error_if(
                new_volume_stack,
                ~jnp.isfinite(new_volume_stack),
                "new_volume_stack is non finite",
            )
        new_particles = eqx.tree_at(
            lambda state: (
                state.volume_stack,
                state.F_stack,
                state.L_stack,
                state.position_stack,
                state.velocity_stack,
            ),
            material_points,
            (
                new_volume_stack,
                new_F_stack,
                new_L_stack,
                new_position_stack,
                new_velocity_stack,
            ),
        )
        
        # jax.debug.print("new_particles.F_stack:{}", new_particles.F_stack)

        return new_solver, new_particles
