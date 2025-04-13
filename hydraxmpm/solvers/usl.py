# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

"""Implementation of the Explicit Update Stress Last (USL) ConstitutiveLaw Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'ConstitutiveLaw point method after 25 years:
    theory, implementation, and applications.'
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Callable, Optional, Self, Tuple

from ..common.types import TypeFloat, TypeInt
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..forces.force import Force
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints

from .mpm_solver import MPMSolver


class USL(MPMSolver):
    """Update Stress Last (USL) ConstitutiveLaw Point Method (MPM) solver."""

    alpha: TypeFloat = eqx.field(default=0.99)

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

    def update(self: Self, step: TypeInt = 0, dt: TypeFloat = 1e-3) -> Self:
        # loading state
        material_points = self.material_points._refresh()

        material_points, forces = self._update_forces_on_points(
            material_points=material_points,
            grid=self.grid,
            forces=self.forces,
            step=step,
            dt=dt,
        )

        # grid quantities are being reset here.
        new_shape_map, grid = self.p2g(
            material_points=material_points, grid=self.grid, dt=dt
        )

        grid, forces = self._update_forces_grid(
            material_points=material_points, grid=grid, forces=forces, step=step, dt=dt
        )

        material_points = self.g2p(
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

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * intr_velocities
            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = scaled_int_force[: self.dim] + scaled_ext_force

            scaled_normal = (intr_shapef_grad * intr_masses).at[: self.dim].get()

            return scaled_mass, scaled_moments, scaled_total_force, scaled_normal

        # note the interactions and shapefunctions are calculated on the
        # p2g to reduce computational overhead.
        (
            new_shape_map,
            (
                scaled_mass_stack,
                scaled_moment_stack,
                scaled_total_force_stack,
                scaled_normal_stack,
            ),
        ) = self.shape_map.vmap_interactions_and_scatter(
            vmap_intr_p2g, material_points, grid
        )

        def sum_interactions(stack, scaled_stack):
            return (
                jnp.zeros_like(stack)
                .at[new_shape_map._intr_hash_stack]
                .add(scaled_stack)
            )

        # sum
        new_mass_stack = sum_interactions(grid.mass_stack, scaled_mass_stack)
        new_moment_stack = sum_interactions(grid.moment_stack, scaled_moment_stack)
        new_force_stack = sum_interactions(
            self.grid.moment_stack, scaled_total_force_stack
        )
        new_normal_stack = sum_interactions(grid.normal_stack, scaled_normal_stack)

        # integrate
        new_moment_nt_stack = new_moment_stack + new_force_stack * self.dt

        return new_shape_map, eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
                state.normal_stack,
            ),
            grid,
            (new_mass_stack, new_moment_stack, new_moment_nt_stack, new_normal_stack),
        )

    def g2p(self, material_points, grid, shape_map, dt) -> MaterialPoints:
        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, _):
            intr_masses = grid.mass_stack.at[intr_hashes].get()
            intr_moments = grid.moment_stack.at[intr_hashes].get()
            intr_moments_nt = grid.moment_nt_stack.at[intr_hashes].get()

            # Small mass cutoff to avoid unphysical large velocities
            intr_vels = jax.lax.cond(
                intr_masses > grid.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments,
            )

            intr_vels_nt = jax.lax.cond(
                intr_masses > grid.small_mass_cutoff,
                lambda x: x / intr_masses,
                lambda x: jnp.zeros_like(x),
                intr_moments_nt,
            )
            intr_delta_vels = intr_vels_nt - intr_vels

            intr_scaled_delta_vels = intr_shapef * intr_delta_vels

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                self._padding,
                mode="constant",
                constant_values=0,
            )

            intr_scaled_velgrad = (
                intr_shapef_grad.reshape(-1, 1) @ intr_vels_nt_padded.reshape(-1, 1).T
            )

            return intr_scaled_delta_vels, intr_scaled_vels_nt, intr_scaled_velgrad

        (
            new_intr_scaled_delta_vel_stack,
            new_intr_scaled_vel_nt_stack,
            new_intr_scaled_velgrad_stack,
        ) = shape_map.vmap_intr_gather(vmap_intr_g2p)

        @partial(jax.vmap, in_axes=0)
        def vmap_particles_update(
            intr_delta_vels_reshaped,
            intr_vels_nt_reshaped,
            intr_velgrad_reshaped,
            p_velocities,
            p_positions,
            p_F,
            p_volumes_orig,
        ):
            """Update particle quantities by summing interaction quantities."""
            p_velgrads_next = jnp.sum(intr_velgrad_reshaped, axis=0)

            delta_vels = jnp.sum(intr_delta_vels_reshaped, axis=0)
            vels_nt = jnp.sum(intr_vels_nt_reshaped, axis=0)

            p_velocities_next = (1.0 - self.alpha) * vels_nt + self.alpha * (
                p_velocities + delta_vels
            )

            p_positions_next = p_positions + vels_nt * dt

            if self.dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0)

            p_F_next = (jnp.eye(3) + p_velgrads_next * dt) @ p_F

            if self.dim == 2:
                p_F_next = p_F_next.at[2, 2].set(1)

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig
            return (
                p_velocities_next,
                p_positions_next,
                p_F_next,
                p_volumes_next,
                p_velgrads_next,
            )

        (
            new_velocity_stack,
            new_position_stack,
            new_F_stack,
            new_volume_stack,
            new_L_stack,
        ) = vmap_particles_update(
            new_intr_scaled_delta_vel_stack.reshape(
                -1, shape_map._window_size, self.dim
            ),
            new_intr_scaled_vel_nt_stack.reshape(-1, shape_map._window_size, self.dim),
            new_intr_scaled_velgrad_stack.reshape(-1, shape_map._window_size, 3, 3),
            material_points.velocity_stack,
            material_points.position_stack,
            material_points.F_stack,
            material_points.volume0_stack,
        )

        return eqx.tree_at(
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
