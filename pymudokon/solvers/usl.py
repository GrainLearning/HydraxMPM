"""Implementation of the Explicit Update Stress Last (USL) Material Point Method (MPM).

References:
    - De Vaucorbeil, Alban, et al. 'Material point method after 25 years:
    theory, implementation, and applications.'
"""

from functools import partial
from typing import List, Tuple
from pyvista import Grid
from typing_extensions import Self

import chex
import jax
import jax.numpy as jnp

# from ..forces.forces import Forces
# from ..materials.material import Material
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from ..partition.grid_stencil_map import GridStencilMap
from ..config.mpm_config import MPMConfig

import equinox as eqx


class USL(eqx.Module):
    """Update Stress Last (USL) Material Point Method (MPM) solver.

    Attributes:
        alpha: FLIP-PIC ratio
        dt: time step of the solver

    """

    alpha: float = eqx.field(static=True, converter=lambda x: float(x))
    dt: float = eqx.field(static=True, converter=lambda x: float(x))
    dim: int = eqx.field(static=True, converter=lambda x: int(x))

    def __init__(
        self, config: MPMConfig, alpha: float = 0.99, dt: float = None, dim: int = 3
    ):
        if config:
            dt = config.dt
            dim = config.dim

        self.dt = dt
        self.alpha = alpha
        self.dim = dim

    def update(
        self: Self,
        prev_particles: Particles,
        prev_nodes: Nodes,
        prev_shapefunctions: ShapeFunction,
        prev_grid: GridStencilMap,
        prev_material_stack: List,
        prev_forces_stack: List,
        step: int = 0,
    ):
        """Perform a single update step of the USL solver."""
        new_particles = prev_particles.refresh()
        new_nodes = prev_nodes.refresh()

        new_grid = prev_grid.partition(new_particles.position_stack)

        new_shapefunctions = prev_shapefunctions.get_shapefunctions(
            new_grid, new_particles
        )

        new_nodes = self.p2g(
            particles=new_particles,
            nodes=new_nodes,
            shapefunctions=new_shapefunctions,
            grid=new_grid,
        )

        # Apply forces here
        new_forces_stack = []
        # for forces in forces_stack:
        #     nodes, forces = forces.apply_on_nodes_moments(
        #         particles=particles,
        #         nodes=nodes,
        #         shapefunctions=shapefunctions,
        #         dt=self.dt,
        #         step=step,
        #     )
        #     new_forces_stack.append(forces)

        new_particles = self.g2p(
            particles=new_particles,
            nodes=new_nodes,
            shapefunctions=new_shapefunctions,
            grid=new_grid,
        )


        new_material_stack = []
        for material in prev_material_stack:
            new_particles, new_material = material.update_from_particles(
                particles=new_particles
            )
            new_material_stack.append(new_material)

        return (
            self,
            new_particles,
            new_nodes,
            new_shapefunctions,
            new_grid,
            new_material_stack,
            new_forces_stack
        )

    def p2g(
        self: Self,
        particles,
        nodes,
        shapefunctions,
        grid,
    ):
        """Particle (MP)  to grid transfer function.

        Procedure is as follows:
        - Gather particle quantities to interactions.
        - Scale masses, moments, and forces by shape functions.
        - Calculate node internal force from scaled stresses, volumes.
        - Sum interaction quantities to nodes.
        """

        def vmap_p2g(p_id, c_id, w_id, carry):
            mass_prev, moment_prev, moment_nt_prev = carry

            mass = particles.mass_stack.at[p_id].get()

            volumes = particles.volume_stack.at[p_id].get()
            velocities = particles.velocity_stack.at[p_id].get()
            ext_forces = particles.force_stack.at[p_id].get()
            stresses = particles.stress_stack.at[p_id].get()

            shapef = shapefunctions.shapef_stack.at[p_id, w_id].get()
            shapef_grad = shapefunctions.shapef_grad_stack.at[p_id, w_id].get()

            scaled_mass = mass * shapef
            scaled_moment = scaled_mass * velocities
            scaled_ext_force = shapef * ext_forces
            scaled_int_force = -1.0 * volumes * stresses @ shapef_grad

            scaled_total_force = (
                scaled_int_force.at[: self.dim].get() + scaled_ext_force
            )

            scaled_moment_nt = scaled_moment + scaled_total_force * self.dt

            return (
                mass_prev + scaled_mass,
                moment_prev + scaled_moment,
                moment_nt_prev + scaled_moment_nt,
            )

        new_mass_stack, new_moments, new_moments_nt_stack = grid.vmap_grid_gather_fori(
            vmap_p2g, (0.0, jnp.zeros(self.dim), jnp.zeros(self.dim))
        )

        return eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.moment_stack,
                state.moment_nt_stack,
            ),
            nodes,
            (new_mass_stack, new_moments, new_moments_nt_stack),
        )

    def g2p(
        self,
        particles: Particles,
        nodes: Nodes,
        shapefunctions: chex.Array,
        grid,
    ):
        """Grid to particle transfer function."""

        padding = (0, 3 - self.dim)

        def vmap_g2p(p_id, c_id, w_id, carry):
            prev_scaled_delta_vel, prev_scaled_vel_nt, prev_scaled_L = carry
            masses = nodes.mass_stack.at[c_id].get()
            moment = nodes.moment_stack.at[c_id].get()
            moment_nt = nodes.moment_nt_stack.at[c_id].get()

            shapef = shapefunctions.shapef_stack.at[p_id, w_id].get()
            shapef_grad = shapefunctions.shapef_grad_stack.at[p_id, w_id].get()

            # Small mass cutoff to avoid unphysical large velocities
            vel = jax.lax.cond(
                masses > nodes.small_mass_cutoff,
                lambda x: x / masses,
                lambda x: jnp.zeros_like(x),
                moment,
            )

            vel_nt = jax.lax.cond(
                masses > nodes.small_mass_cutoff,
                lambda x: x / masses,
                lambda x: jnp.zeros_like(x),
                moment_nt,
            )
            delta_vel = vel_nt - vel

            scaled_delta_vel = shapef * delta_vel

            scaled_vel_nt = shapef * vel_nt

            # Pad velocities for plane strain
            vel_nt_padded = jnp.pad(
                vel_nt,
                padding,
                mode="constant",
                constant_values=0,
            )

            scaled_L = shapef_grad.reshape(-1, 1) @ vel_nt_padded.reshape(-1, 1).T

            carry = (
                prev_scaled_delta_vel + scaled_delta_vel,
                prev_scaled_vel_nt + scaled_vel_nt,
                prev_scaled_L + scaled_L,
            )
            return carry

        delta_vel_stack, vel_nt_stack, L_stack = grid.vmap_grid_scatter_fori(
            vmap_g2p, (jnp.zeros(self.dim), jnp.zeros(self.dim), jnp.zeros((3, 3)))
        )

        @jax.vmap
        def vmap_update_particles(
            vol0, prev_F, prev_pos, prev_vel, delta_vel, vel_nt, next_L
        ):
            next_vel = (1.0 - self.alpha) * vel_nt + self.alpha * (prev_vel + delta_vel)
            next_pos = prev_pos + vel_nt * self.dt

            if self.dim == 2:
                next_L = next_L.at[2, 2].set(0)

            next_F = (jnp.eye(3) + next_L * self.dt) @ prev_F

            if self.dim == 2:
                next_F = next_F.at[2, 2].set(1)

            vol_next = jnp.linalg.det(next_F) * vol0

            return vol_next, next_F, next_L, next_pos, next_vel

        (
            next_volume_stack,
            next_F_stack,
            next_L_stack,
            next_position_stack,
            next_velocity_stack,
        ) = vmap_update_particles(
            particles.volume0_stack,
            particles.F_stack,
            particles.position_stack,
            particles.velocity_stack,
            delta_vel_stack,
            vel_nt_stack,
            L_stack,
        )

        return eqx.tree_at(
            lambda state: (
                state.volume_stack,
                state.F_stack,
                state.L_stack,
                state.position_stack,
                state.velocity_stack,
            ),
            particles,
            (
                next_volume_stack,
                next_F_stack,
                next_L_stack,
                next_position_stack,
                next_velocity_stack,
            ),
        )
