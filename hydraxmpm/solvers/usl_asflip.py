# # Copyright (c) 2024, Retiefasuarus
# # SPDX-License-Identifier: BSD-3-Clause
# #
# # Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# # -*- coding: utf-8 -*-
"""
Explanation:

    This module implements the USL-AFLIP solver for the Material Point Method (MPM).

    The USL-AFLIP has a state `USLAFLIPState` and solver logic `USLAFLIP`.

    Additional features:
        - **AFLIP/APIC**: Preserves angular momentum and reduces dissipation compared to PIC.
        - **CFL Limiting**: Clamps particle velocity to a fraction of cell_size/dt.
        - **Dynamic Alpha**: Blends FLIP/PIC based on particle support to prevent instability in empty cells.
        - **Position Correction**: Uses separate velocity fields for position update to avoid the "positional trap".


    References:
    - Fei, Yun, et al. "Revisiting integration in the material point method: a scheme for easier separation and less dissipation."
    - Jiang, Chenfanfu, et al. "The affine particle-in-cell method."

"""

import equinox as eqx

from typing import Self, Tuple

import jax.numpy as jnp

from ..grid.grid import GridDomain
from ..material_points.material_points import MaterialPointState

from .solver import BaseSolver, BaseSolverState

from ..common.simstate import SimState

from ..forces.force import Force

from .coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw

from ..forces.sdf_collider import apply_frictional_contact

from ..shapefunctions.mapping import InteractionCache

from ..sdf.sdfobject import SDFObjectBase

from jaxtyping import Float, Array

from typing import Tuple, Optional

from .usl import USLSolver
import jax




def apply_fbar_correction(
    F_inc: Float[Array, "num_points 3 3"],
    intr_cache: InteractionCache,
    intr_mass_stack: Float[Array, "num_intr"], # Gathered grid mass (115200,)
    num_grid_nodes: int,
    num_points: int,
    dim: int
) -> Float[Array, "num_points 3 3"]:
    """
    Applies a simple F-bar volumetric averaging to the deformation gradient increment.
    """
    # 1. Calculate local volume change (Shape: [12800])
    p_J_inc_local = jnp.linalg.det(F_inc)
    
    # 2. EXPAND J to interaction level (Shape: [115200])
    # This maps the 12800 particle values to the 115200 interaction slots
    intr_J_inc = p_J_inc_local.at[intr_cache.point_ids].get()
    
    # 3. Scatter volume change to nodes
    # Now all arrays are (115200,) and the multiplication will work
    weighted_J = intr_cache.shape_vals * intr_J_inc * intr_mass_stack
    node_weight = intr_cache.shape_vals * intr_mass_stack
    
    node_J_num = jnp.zeros((num_grid_nodes,)).at[intr_cache.node_hashes].add(weighted_J)
    node_J_den = jnp.zeros((num_grid_nodes,)).at[intr_cache.node_hashes].add(node_weight)
    
    # Compute nodal average J
    node_J_avg = node_J_num / (node_J_den + 1e-12)
    
    # 4. Gather average J back to particles (Shape: [12800])
    # We take the nodal averages, map them to interactions, and sum them up for the particle
    p_J_avg = jnp.zeros((num_points,)).at[intr_cache.point_ids].add(
        intr_cache.shape_vals * node_J_avg.at[intr_cache.node_hashes].get()
    )
    
    # 5. Rescale F_inc (Both arrays are now [12800])
    scale_factor = jnp.power(p_J_avg / (p_J_inc_local + 1e-12), 1.0 / 3.0)
    F_rescaled = F_inc * scale_factor[:, None, None]
    
    # if dim == 2:
    #     F_rescaled = F_inc.at[:, :2, :2].multiply(scale_factor[:, None, None])
    # else:
    #     F_rescaled = F_inc * scale_factor[:, None, None]
        
    return F_rescaled

class USLAFLIPState(BaseSolverState):
    """
    State required for the AFLIP/APIC solver.

    Attributes:
        Bp_stack: Affine momentum matrices for each material point.
    """

    Bp_stack: Float[Array, "num_points 3 3"]


class USLAFLIP(USLSolver):
    """
    Update Stress Last (USL) Solver with Affine-FLIP (AFLIP/APIC) transfer.

    Position Correction (Separable S-FLIP) uses beta_p which mixes the FLIP and PIC velocities
    specifically for the position update.

    Dynamic Alpha reduces the FLIP component when particles have low grid support.

    CFL Limiting clamps particle velocities to a fraction of cell_size/dt to ensure stability.

    Attributes:
        alpha: Blending factor between PIC and FLIP updates
            (0.0 = pure PIC, 1.0 = pure FLIP; default 0.99).
        use_dynamic_alpha: Enables dynamic alpha scaling based on particle support (default True).
        cfl_limit: Max fraction of cell_size per step (default 0.5)
        beta_min: Mixes FLIP/PIC based on particle support (minimum), for position update (default 1.0).
        beta_max: Mixes FLIP/PIC based on particle support (maximum), for position update (default 0.5).
        small_mass_cutoff: Prevents updates from grid to particles if masses are too small (default 1e-7).
    """

    # FLIP/PIC blending ratio
    alpha: float = eqx.field(static=True)

    # Position Correction (Separable S-FLIP)
    beta_min: float = eqx.field(static=True, default=0.0)
    beta_max: float = eqx.field(static=True, default=1.0)
    rho_0: float = eqx.field(static=True, default=1000.0)

    # CFL Condition
    cfl_limit: float = eqx.field(static=True, default=0.5)
    small_mass_cutoff: float = eqx.field(static=True, default=1e-7)

    # Logic operations
    couplings: Tuple[BodyCoupling, ...]
    constitutive_laws: Tuple[ConstitutiveLaw, ...]
    forces: Tuple[Force, ...]
    sdf_logics: Tuple[SDFObjectBase, ...]
    grid_domains: Tuple[GridDomain, ...] = eqx.field(static=True)

    active_p_ids: Tuple[int, ...] = eqx.field(static=True)
    active_g_ids: Tuple[int, ...] = eqx.field(static=True)
    
    # Use exponential map for deformation gradient update instead of linearized update
    exponential_F: bool = eqx.field(static=True, default=False)
    
    def create_state(self, mp_state) -> Self:
        """Creates empty state with affine matrices"""
        return USLAFLIPState(Bp_stack=jnp.zeros((mp_state.num_points, 3, 3)))

    def __init__(
        self,
        *,
        grid_domains: Tuple[GridDomain, ...],
        constitutive_laws: Tuple[Optional[ConstitutiveLaw], ...],
        couplings: Tuple[BodyCoupling, ...] = (),
        forces: Tuple[Optional[Force], ...] = (),
        sdf_logics: Optional[Tuple[SDFObjectBase, ...]] = (),
        alpha=0.99,
        beta_min: float = 0.0,
        beta_max: float = 1.0,
        rho_0: float = 1000.0,
        small_mass_cutoff: float = 1.0e-7,
        cfl_limit: float = 0.5,
        exponential_F: bool = False,
    ):
        # FLIP/ PIC
        self.alpha = alpha

        # Seperable prevent positional trap
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.rho_0 = rho_0

        # Stability
        self.small_mass_cutoff = small_mass_cutoff
        self.cfl_limit = cfl_limit

        # Deformation gradient update
        self.exponential_F = exponential_F

        # logic operations
        self.constitutive_laws = constitutive_laws
        self.couplings = couplings
        self.forces = forces
        self.sdf_logics = sdf_logics

        p_set = sorted(list(set(c.p_idx for c in couplings)))
        g_set = sorted(list(set(c.g_idx for c in couplings)))

        self.active_p_ids = tuple(p_set)
        self.active_g_ids = tuple(g_set)

        self.grid_domains = grid_domains

    def _p2g(self, world, mechanics, sim_cache, dt, time):
        """Particle to Grid Transfer (AFLIP/APIC)."""

        for force in self.forces:
            world, mechanics, sim_cache = force.apply_pre_p2g(
                world,
                mechanics,
                sim_cache,
                self.sdf_logics,
                self.couplings,
                self.grid_domains,
                dt,
                time,
            )

        grids = list(sim_cache.grids)
        mp_states = list(world.material_points)
        grid_domains = self.grid_domains

        solver_states = list(mechanics.solvers)

        for c in self.couplings:
            # Ignore non-MPM couplings
            if c.skip_mpm_logic:
                continue
            mp_state = mp_states[c.p_idx]
            grid_cache = grids[c.g_idx]
            grid_domain = grid_domains[c.g_idx]
            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]
            solver_state = solver_states[c.s_idx]

            # ==============================
            # Gather to interaction space
            # =============================
            # Gather material point data to interaction space
            intr_masses_stack = mp_state.mass_stack.at[intr_cache.point_ids].get()
            intr_velocities_stack = mp_state.velocity_stack.at[
                intr_cache.point_ids
            ].get()
            intr_ext_forces_stack = mp_state.force_stack.at[intr_cache.point_ids].get()
            intr_stress_stack = mp_state.stress_stack.at[intr_cache.point_ids].get()

            intr_volume0_stack = mp_state.volume0_stack.at[intr_cache.point_ids].get()

            # ==============================
            #  MLS shape functions gradients
            # =============================


            # Get kernel inertia tensor inverse
            # for quadratic B-spline kernels
            if c.shape_map.shapefunction == "quadratic":
                Dp_inv = 4.0 / (grid_domain.cell_size**2)
            elif c.shape_map.shapefunction == "cubic":
                Dp_inv = 3.0 / (grid_domain.cell_size**2)


            # Relative distance from particle to nodes in world coordinates
            #  
            # We multiply by cell_size 
            # intr_cache.rel_dist = (x_p- x_i) / cell size
            # we need (x_i - x_p)
            x_i_m_x_p = -1.0 * intr_cache.rel_dist * grid_domain.cell_size

            grad_shape_vals = intr_cache.shape_vals[:, None] * Dp_inv * x_i_m_x_p
            # grad_shape_vals = intr_cache.shape_grads
            # ==============================
            #  CPIC shape function masking
            # =============================
            # Apply mask to seperate compatible and non compatible 
            # interactions based on boundary
            compatible_shape_vals = intr_cache.shape_vals * intr_cache.cpic_mask
            compatible_grad_shape_vals = grad_shape_vals * intr_cache.cpic_mask[:, None]

            # ==============================
            # APIC velocity split
            # =============================
             
            # Get Bp affine matrix 
            intr_Bp = solver_state.Bp_stack.at[intr_cache.point_ids].get()

            # compression positive sign
            v_affine = jnp.einsum("nij,nj->ni", intr_Bp, x_i_m_x_p)
            
            total_intr_velocities_stack = (
                intr_velocities_stack - v_affine[:, : grid_domain.dim]
            )


            # ==============================
            #  MPM internal and external forces
            # =============================
            #  Mass contribution
            # m_i = Σ_p m_p N_ip
            weighted_mass_stack = compatible_shape_vals * intr_masses_stack

            # Momentum contribution
            # (m v)_i = Σ_p m_p v_p N_ip
            #  affine term is included already
            weighted_moment_stack = weighted_mass_stack[:, None] * (
                total_intr_velocities_stack
            )

            # External forces contributions
            # f_i,ext = Σ_p f_p N_ip
            weighted_ext_force_stack = (
                compatible_shape_vals[:, None] * intr_ext_forces_stack
            )
            # Internal forces contribution
            # f_i,int = Σ_p V_p P_p ∇N_ip
            # Kirchhoff stress is used
            # Sign is for compression positive +
            weighted_intern_force_stack =  intr_volume0_stack[:, None] *(intr_stress_stack @ compatible_grad_shape_vals[..., None]).squeeze(
                -1
            )[:, : grid_cache.dim]

            # Total force contribution
            # f_i = f_i,ext + f_i,int
            total_intr_force = weighted_intern_force_stack + weighted_ext_force_stack

            # ==============================
            #  Scatter to grid
            # =============================
            grid_mass_stack = grid_cache.mass_stack.at[intr_cache.node_hashes].add(
                weighted_mass_stack
            )
            grid_moment_stack = grid_cache.moment_stack.at[intr_cache.node_hashes].add(
                weighted_moment_stack
            )
            grid_force_stack = (
                jnp.zeros_like(grid_cache.moment_stack)
                .at[intr_cache.node_hashes]
                .add(total_intr_force)
            )

            grids[c.g_idx] = eqx.tree_at(
                lambda s: (s.mass_stack, s.moment_stack, s.force_stack),
                grid_cache,
                (
                    grid_mass_stack,
                    grid_moment_stack,
                    grid_force_stack,
                ),
            )

        sim_cache = eqx.tree_at(
            lambda s: s.grids,
            sim_cache,
            tuple(grids),
        )
        return world, mechanics, sim_cache

    def _g2p(
        self,
        world,
        mechanics,
        sim_cache,
        dt,
        time,
    ):
        """Grid to Particle Transfer (AFLIP/APIC with Position Correction and Dynamic Alpha)."""
        # Apply forces hook 4 to modify grid moments, e.g., grid contact

        grids = list(sim_cache.grids)
        mp_states = list(world.material_points)
        solver_states = list(mechanics.solvers)

        for c in self.couplings:
            if c.skip_mpm_logic:
                continue
            grid_cache = grids[c.g_idx]
            grid_domain = self.grid_domains[c.g_idx]
            mp_state = mp_states[c.p_idx]
            intr_cache = sim_cache.interactions[(c.p_idx, c.g_idx)]
            solver_state = solver_states[c.p_idx]


            # ==============================
            # Gather to interaction space
            # =============================
            # Gather Grid data to interaction space
            intr_mass_stack = grid_cache.mass_stack.at[intr_cache.node_hashes].get()
            intr_momement_stack = grid_cache.moment_stack.at[
                intr_cache.node_hashes
            ].get()
            intr_momement_nt_stack = grid_cache.moment_nt_stack.at[
                intr_cache.node_hashes
            ].get()
            
            # ==============================
            # MLS shape functions gradients
            # =============================

            # Get kernel inertia tensor inverse
            # for quadratic B-spline kernels
            if c.shape_map.shapefunction == "quadratic":
                Dp_inv = 4.0 / (grid_domain.cell_size**2)
            elif c.shape_map.shapefunction == "cubic":
                Dp_inv = 3.0 / (grid_domain.cell_size**2)

            # Relative distance from particle to nodes in world coordinates
            #  (x_i - x_p)
            # We multiply by cell_size 
            x_i_m_x_p = -1.0 * intr_cache.rel_dist * grid_domain.cell_size
            
            # grad_shape_vals = intr_cache.shape_grads
            grad_shape_vals = intr_cache.shape_vals[:, None] * Dp_inv * x_i_m_x_p

            # ==============================
            # MPM get node (safe) velocities
            # =============================
            # Small mass cutoff to prevent instabilities
            safe_masses = jnp.where(
                intr_mass_stack > self.small_mass_cutoff, intr_mass_stack, 1.0
            )[:, None]
            mask = (intr_mass_stack > self.small_mass_cutoff)[:, None]


            # immediately after the P2G transfer
            # particle velocities projected on grid
            intr_vels = jnp.where(mask, intr_momement_stack / safe_masses, 0.0)

            # velocities on grid after integration of forces
            # This is the velocity we want to use for the FLIP update to preserve momentum changes from forces 
            intr_vels_nt = jnp.where(mask, intr_momement_nt_stack / safe_masses, 0.0)

            # ==============================
            # MLS affine velocity field
            # =============================
            intr_vels_nt_B_p = intr_vels_nt
            
            # ==============================
            # CPIC compatible and non compatible corrections
            # =============================

            # min_dist_to_wall = jnp.full((mp_state.num_points,), 1e9)

            if len(self.sdf_logics) > 0:
            #     # Stack distances
                num_sdfs = len(self.sdf_logics)
                dists_stack = jnp.stack([sim_cache.mp_geoms[(c.p_idx, s)].dists.squeeze() for s in range(num_sdfs)])
                norms_stack = jnp.stack([sim_cache.mp_geoms[(c.p_idx, s)].normals for s in range(num_sdfs)])
                vels_stack  = jnp.stack([sim_cache.mp_geoms[(c.p_idx, s)].wall_vels for s in range(num_sdfs)])
                fric_stack = jnp.stack([sim_cache.mp_geoms[(c.p_idx, s)].friction for s in range(num_sdfs)])

            #     # Find closest
                closest_idx = jnp.argmin(dists_stack, axis=0, keepdims=True)
                # closest_idx= jnp.atleast_1d([0])
                # Store min dist for ASFLIP safety
                min_dist_to_wall = jnp.take_along_axis(dists_stack, closest_idx, axis=0).squeeze(0)

                p_normal_best = jnp.take_along_axis(norms_stack, closest_idx[..., None], axis=0).squeeze(0)
                p_wall_vel_best = jnp.take_along_axis(vels_stack, closest_idx[..., None], axis=0).squeeze(0)
                p_fric_best = jnp.take_along_axis(fric_stack, closest_idx, axis=0).squeeze(0)


            #     p_vel_ghost = jax.vmap(apply_frictional_contact, in_axes=(0, 0, 0, 0, 0, None, None, None))(
            #             mp_state.velocity_stack,
            #             min_dist_to_wall,
            #             p_normal_best,
            #             p_wall_vel_best,
            #             p_fric_best,
            #             dt,
            #             0.0025/4,
            #             0.0
            #         )
            #     intr_vels_ghost = p_vel_ghost.at[intr_cache.point_ids].get()

            #     intr_vels_ghost_B_p = mp_state.velocity_stack.at[intr_cache.point_ids].get()

            #     intr_vels_ghost = intr_vels_ghost.at[intr_cache.point_ids].get()
            #     intr_vels_ghost_B_p = intr_vels_ghost_B_p.at[intr_cache.point_ids].get()
            # else:
            #     intr_vels_ghost = jnp.zeros_like(intr_vels)
            #     intr_vels_ghost_B_p = mp_state.velocity_stack.at[intr_cache.point_ids].get()

            mask = intr_cache.cpic_mask[:, None]
            # jax.debug.print("cpic_mask unique values: {p}", p=jnp.unique(intr_cache.cpic_mask, size=intr_cache.cpic_mask.shape[0]))
            # jax.debug.print("cpic_mask shape: {p}", p=intr_cache.cpic_mask.shape)
            # jax.debug.print("cpic_mask non-zero count: {p}", p=jnp.sum(intr_cache.cpic_mask > 0))
            # intr_vels = intr_vels*mask  + intr_vels_ghost*(1-mask)
            # intr_vels_nt = intr_vels_nt*mask  + intr_vels_ghost*(1-mask)
            # intr_vels_nt_B_p = intr_vels_nt_B_p * mask  + intr_vels_ghost_B_p * (1 - mask)



            # intr_vels = intr_vels*mask + intr_vels_ghost*(1-mask)
            # + intr_vels_ghost*(1-mask)
            # Use particle velocity for non-compatible nodes 
            # 
            
            # debug
            # intr_vels_nt = intr_vels_nt*mask + intr_vels_nt*(1.0-mask)
            # intr_vels = intr_vels*mask + intr_vels*(1.0-mask)
            # intr_vels_nt_B_p = intr_vels_nt_B_p*mask + intr_vels_nt_B_p*(1.0-mask)
            # debug

            # intr_vels = intr_vels
            # intr_vels_nt = intr_vels_nt

            # # Use particle velocity for non-compatible nodes 
                    

            weighted_vels = intr_cache.shape_vals[:, None] * intr_vels
            weighted_vels_nt = intr_cache.shape_vals[:, None] * intr_vels_nt

            # ==============================
            # MLS affine matrix
            # =============================
            padding = (0, 3 - grid_cache.dim)
            intr_vels_nt_B_p_3d = jnp.pad(intr_vels_nt_B_p, ((0, 0), padding))
   
            # MLS compatible Bp term
            # compression positive
            # L=−∇v
            weighted_Bp_term = -1.0 * jnp.einsum(
                "ij,ik->ijk", intr_vels_nt_B_p_3d, grad_shape_vals
            )

            # ==============================
            # Scatter to particles
            # =============================
            p_vel = (
                jnp.zeros((mp_state.num_points, grid_cache.dim))
                .at[intr_cache.point_ids]
                .add(weighted_vels)
            )

            p_vel_nt = (
                jnp.zeros((mp_state.num_points, grid_cache.dim))
                .at[intr_cache.point_ids]
                .add(weighted_vels_nt)
            )

            # Interpolated affine matrix
            p_Bp = (
                jnp.zeros((mp_state.num_points, 3, 3))
                .at[intr_cache.point_ids]
                .add(weighted_Bp_term)
            )
            
            # ==============================
            # Material point velocity update with FLIP/PIC blending
            # ==============================

            # velocity fluctuation term for FLIP update
            vel_adj = mp_state.velocity_stack - p_vel
            
            
            # mp.specific_volume_stack = mp_state.volume_stack / mp_state.mass_stack
            
            rho_p = 2650.0
            rho_stack = mp_state.mass_stack / (mp_state.volume_stack + 1e-12)
            
            phi_p = rho_stack/rho_p
            
            # compressing value approaches zero
            # decompressing means phi_p approaches phi_max
            

            phi_max = 0.35
            
            phi_min = 0.25
            
            alpha = jnp.clip((phi_max - phi_p)/(phi_min - phi_max), 0.0, 0.9)
            # ratio = 1- phi_max/phi_p
            # jax.debug.print("phi_p max {a} min {i} ",a=jnp.max(phi_p), i=jnp.min(phi_p))
            
            # alpha = jnp.clip(ratio, 0.0, 1.0)

            alpha = jnp.ones_like(phi_p) * 0.0
            
            p_velocity_next = p_vel_nt + alpha[:, None] * vel_adj

            # p_velocity_next = p_vel_nt + self.alpha * vel_adj

            # ==============================
            # Material point velocity CFL clamping
            # ==============================
            # This is a regularization to prevent particles crossing >50% of a cell in one step
            max_speed = self.cfl_limit * grid_domain.cell_size / dt
            speed = jnp.linalg.norm(p_velocity_next, axis=1, keepdims=True)
            clamp_factor = jnp.minimum(1.0, max_speed / (speed + 1e-12))
            p_velocity_next = p_velocity_next * clamp_factor

            # ==============================
            # Material point deformation update
            # ==============================
            # in MLS MPM affine term is taken as velocity gradient () for MLS MPM
            # We use L=-p_Bp, compression positive sign

            if grid_cache.dim == 2:
                p_Bp = p_Bp.at[:, 2, 2].set(0.0)

            # 1. Kinematic increment
            if self.exponential_F:
                F_inc = jax.scipy.linalg.expm(-p_Bp * dt)
            else:
                F_inc = jnp.eye(3) - p_Bp * dt

            if grid_cache.dim == 2:
                F_inc = F_inc.at[:, 2, 2].set(1.0)

            intr_mass_stack = grid_cache.mass_stack.at[intr_cache.node_hashes].get()

            
            # optionally store deformation gradient 
            if mp_state.F_stack is not None:
                F_stack = jnp.einsum("ijk,ikl->ijl", F_inc, mp_state.F_stack)
            else:
                F_stack = None

            # ==============================
            # Material point volume update
            # ==============================

            J_inc = jnp.linalg.det(F_inc)
            p_volume_next = mp_state.volume_stack * J_inc

           # ==============================
           # Separable S-FLIP position update
           # ==============================
           # Note S-FLIP requires check for boundary safety.. 
           # Otherwise layering will occur because correction term pushes particles away from
           # boundaries during expansion
           # We need to add safety check

            # Jp = p_volume_next/mp_state.volume0_stack

            # beta_p = jnp.where(Jp < 1.0, self.beta_min, self.beta_max)


            # correction_term = self.alpha * beta_p[:, None] * vel_adj
            
            # # correction_term = 0.0
            # # asflip_safety = jnp.where(min_dist_to_wall < grid_domain.cell_size, 0.0, 1.0)
            # # correction_term = correction_term * asflip_safety[:, None]

            # p_position_next = mp_state.position_stack + dt * (
            #     p_vel_nt + correction_term
            # )

            p_position_next = mp_state.position_stack + dt * (
                p_vel_nt
            )
            
            if len(self.sdf_logics) > 0:
                penalty_stiffness = jnp.array([10_000.0])
                penetration = jnp.minimum(min_dist_to_wall, 0.0)
                v_penalty = - penetration[:, None] * p_normal_best * penalty_stiffness
                p_velocity_next = p_velocity_next + v_penalty


            mp_states[c.p_idx] = eqx.tree_at(
                lambda s: (
                    s.velocity_stack,
                    s.position_stack,
                    s.volume_stack,
                    s.F_inc_stack,
                    s.F_stack,
                ),
                mp_state,
                (p_velocity_next, p_position_next, p_volume_next, F_inc, F_stack),
            )

            solver_states[c.p_idx] = eqx.tree_at(
                lambda s: s.Bp_stack, solver_state, p_Bp
            )

        world = eqx.tree_at(
            lambda w: (w.material_points,),
            world,
            (tuple(mp_states),),
        )

        mechanics = eqx.tree_at(
            lambda w: (w.solvers,),
            mechanics,
            (tuple(solver_states),),
        )
        return world, mechanics, sim_cache
