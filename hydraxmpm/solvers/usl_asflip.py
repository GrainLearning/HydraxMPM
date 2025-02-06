from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import Callable, Optional, Self, Tuple

from ..common.types import (
    TypeFloat,
    TypeFloatMatrix3x3,
    TypeFloatMatrix3x3PStack,
    TypeInt,
)
from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..forces.force import Force
from ..grid.grid import Grid
from ..material_points.material_points import MaterialPoints
from ..solvers.config import Config
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

    def __init__(
        self,
        config: Config,
        material_points: MaterialPoints,
        grid: Grid,
        constitutive_laws: Optional[
            Tuple[ConstitutiveLaw, ...] | ConstitutiveLaw
        ] = None,
        forces: Optional[Tuple[Force, ...]] = None,
        callbacks: Optional[Tuple[Callable, ...] | Callable] = None,
        alpha: Optional[TypeFloat] = 1.0,
        phi_c: TypeFloat = 0.5,
        beta_min: TypeFloat = 0.0,
        beta_max: TypeFloat = 0.0,
        **kwargs,
    ):
        super().__init__(
            config=config,
            material_points=material_points,
            grid=grid,
            constitutive_laws=constitutive_laws,
            forces=forces,
            callbacks=callbacks,
            **kwargs,
        )

        self.alpha = alpha
        self.phi_c = phi_c
        self.beta_min = beta_min
        self.beta_max = beta_max

        if self.config.shapefunction != "cubic":
            raise NotImplementedError("Only cubic shapefunctions supported with ASFLIP")

        Dp = kwargs.get("Dp", None)
        Dp_inv = kwargs.get("Dp_inv", None)
        Bp_stack = kwargs.get("Bp_stack", None)

        if Dp is None:
            self.Dp = (
                (1.0 / 3.0) * self.grid.cell_size * self.grid.cell_size * jnp.eye(3)
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

    def update(self: Self, step: TypeInt = 0) -> Self:
        material_points = self.material_points._refresh()

        material_points, forces = self._update_forces_on_points(
            material_points=material_points,
            grid=self.grid,
            forces=self.forces,
            step=step,
        )

        self, grid = self.p2g(material_points=material_points, grid=self.grid)

        grid, forces = self._update_forces_grid(
            material_points=material_points, grid=grid, forces=forces, step=step
        )

        self, material_points = self.g2p(material_points=material_points, grid=grid)

        material_points, constitutive_laws = self._update_constitutive_laws(
            material_points, self.constitutive_laws
        )

        return eqx.tree_at(
            lambda state: (
                state.material_points,
                state.grid,
                state.constitutive_laws,
                state.forces,
            ),
            self,
            (material_points, grid, constitutive_laws, forces),
        )

    def p2g(self, material_points, grid):
        def vmap_intr_p2g(point_id, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = material_points.mass_stack.at[point_id].get()
            intr_volumes = material_points.volume_stack.at[point_id].get()
            intr_velocities = material_points.velocity_stack.at[point_id].get()
            intr_ext_forces = material_points.force_stack.at[point_id].get()
            intr_stresses = material_points.stress_stack.at[point_id].get()

            intr_Bp = self.Bp_stack.at[point_id].get()  # APIC affine matrix

            affine_velocity = (
                intr_Bp @ jnp.linalg.inv(self.Dp)
            ) @ intr_dist  # intr_dist is 3D

            scaled_mass = intr_shapef * intr_masses
            scaled_moments = scaled_mass * (
                intr_velocities + affine_velocity.at[: self.config.dim].get()
            )

            scaled_ext_force = intr_shapef * intr_ext_forces
            scaled_int_force = -1.0 * intr_volumes * intr_stresses @ intr_shapef_grad

            scaled_total_force = (
                scaled_int_force.at[: self.config.dim].get() + scaled_ext_force
            )

            scaled_normal = (intr_shapef_grad * intr_masses).at[: self.config.dim].get()

            return scaled_mass, scaled_moments, scaled_total_force, scaled_normal

        # note the interactions and shapefunctions are calculated on the
        # p2g to reduce computational overhead.
        (
            new_self,
            (
                scaled_mass_stack,
                scaled_moment_stack,
                scaled_total_force_stack,
                scaled_normal_stack,
            ),
        ) = self.vmap_interactions_and_scatter(vmap_intr_p2g)

        def sum_interactions(stack, scaled_stack):
            return jnp.zeros_like(stack).at[new_self._intr_hash_stack].add(scaled_stack)

        # sum
        new_mass_stack = sum_interactions(grid.mass_stack, scaled_mass_stack)
        new_moment_stack = sum_interactions(grid.moment_stack, scaled_moment_stack)
        new_force_stack = sum_interactions(
            self.grid.moment_stack, scaled_total_force_stack
        )
        new_normal_stack = sum_interactions(grid.normal_stack, scaled_normal_stack)

        nodes_moment_nt_stack = new_moment_stack + new_force_stack * self.config.dt

        return new_self, eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
                state.normal_stack,
            ),
            grid,
            (new_mass_stack, new_moment_stack, nodes_moment_nt_stack, new_normal_stack),
        )

    def g2p(self, material_points, grid) -> Tuple[Self, MaterialPoints]:
        def vmap_intr_g2p(intr_hashes, intr_shapef, intr_shapef_grad, intr_dist):
            intr_masses = grid.mass_stack.at[intr_hashes].get()
            intr_moments = grid.moment_stack.at[intr_hashes].get()
            intr_moments_nt = grid.moment_nt_stack.at[intr_hashes].get()

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

            intr_scaled_vels_nt = intr_shapef * intr_vels_nt

            intr_delta_vels = intr_vels_nt - intr_vels
            intr_scaled_delta_vels = intr_shapef * intr_delta_vels

            # Pad velocities for plane strain
            intr_vels_nt_padded = jnp.pad(
                intr_vels_nt,
                self.config._padding,
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
            )

        (
            new_intr_scaled_delta_vel_stack,
            new_intr_scaled_vel_nt_stack,
            new_intr_scaled_velgrad_stack,
            new_intr_Bp_stack,
        ) = self.vmap_intr_gather(vmap_intr_g2p)

        @partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0))
        def vmap_particles_update(
            intr_delta_vels_reshaped,
            intr_vels_nt_reshaped,
            intr_velgrad_reshaped,
            intr_Bp,
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

            if self.config.dim == 2:
                p_Bp_next = p_Bp_next.at[2, 2].set(0.0)

            T = self.alpha * (p_velocities + delta_vels - vels_nt)
            p_velocities_next = vels_nt + T

            if self.config.dim == 2:
                p_velgrads_next = p_velgrads_next.at[2, 2].set(0.0)

            p_F_next = (jnp.eye(3) + p_velgrads_next * self.config.dt) @ p_F

            if self.config.dim == 2:
                p_F_next = p_F_next.at[2, 2].set(1)

            p_volumes_next = jnp.linalg.det(p_F_next) * p_volumes_orig

            # solid volume fraction
            phi = p_volumes_orig / p_volumes_next  # mass is constant

            Beta_p = jax.lax.cond(
                phi < self.phi_c, lambda: self.beta_max, lambda: self.beta_min
            )

            vel_update = vels_nt + Beta_p * T

            p_positions_next = p_positions + vel_update * self.config.dt

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
                -1, self._window_size, self.config.dim
            ),
            new_intr_scaled_vel_nt_stack.reshape(
                -1, self._window_size, self.config.dim
            ),
            new_intr_scaled_velgrad_stack.reshape(-1, self._window_size, 3, 3),
            new_intr_Bp_stack.reshape(-1, self._window_size, 3, 3),
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

        return new_solver, new_particles
