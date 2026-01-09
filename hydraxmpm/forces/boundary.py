# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explaination:
    This module contains boundary force implementations.

    Currently includes planarBoundaries (reflective/frictional) boundary conditions
    on planes for both 2D and 3D.

    The implementation uses simple signed distance checks and velocity adjustments

"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple, Optional, Any

from matplotlib.pylab import indices

from .force import Force, BaseForceState

from ..grid.grid import GridState 


class PlanarBoundaries(Force):
    """
    Logic for planar boundary conditions applied on background grid nodes.

    Attributes:
        points: Points on the boundary planes.
        normals: Normals of the boundary planes.
        frictions: Friction coefficients for each boundary plane (0.0 = Slip, large = Stick).
        g_idx_list: List of grid indices to apply the boundary on.
        f_idx: Index of the boundary force state in the force states.
        gap: Margin of safety distance for collision detection.

    """
    # TODO, do we need a special class for this 
    # or cant we just use SDFCollider with planes?
    
    # physics
    points: Float[Array, "num_walls dim"] 
    normals: Float[Array, "num_walls dim"] 
    frictions: Float[Array, "num_walls"]
    
    # coupling
    g_idx_list: list[int] = eqx.field(static=True)
    f_idx: int = eqx.field(static=True)

    # Margin of safety distance
    gap: float = eqx.field(static=True)

    def __init__(
        self, 
        origin: tuple | Float[Array, "dim"],
        end:  tuple | Float[Array, "dim"],
        frictions: float | Float[Array, "num_walls"]  = 0.0,
        f_idx: int = 0,
        g_idx_list: list[int] = None,
        gap: float = 1e-4 
    ):
        """
        Initialize method for planar boundaries.

        Origin and end define the bounding box of the domain.

        Not frictions can be a single array for all walls,
        or an array defining friction per wall, in the order of
        2D: 4 walls (left, right, bottom, top)
        3D: 6 walls (left, right, bottom, top, front, back)

        """
        if g_idx_list is None:
            g_idx_list = [0] 

        dim = jnp.array(origin).shape[0]
        self.gap = gap

        o = jnp.array(origin)
        e = jnp.array(end)

        if dim == 2:
            # Define 4 walls in 2D
            o_x, o_y = origin
            e_x, e_y = end
            points = [
                o,
                o,
                e,
                e
            ]
            normals = [(0.0, 1.0), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0)]

        elif dim == 3:
            # Define 6 walls in 3D
            # o_x, o_y, o_z = origin
            # e_x, e_y, e_z = end
            points = [
                o,              # Bottom (y=0)
                o,              # Left (x=0)
                o,              # Back (z=0)
                e,              # Right (x=10)
                e,              # Top (y=10)
                e               # Front (z=10)
            ]
            normals = [
                (0.0, 1.0, 0.0),  # Bottom (Up)
                (1.0, 0.0, 0.0),  # Left (Right)
                (0.0, 0.0, 1.0),  # Back (Forward)
                (-1.0, 0.0, 0.0), # Right (Left)
                (0.0, -1.0, 0.0), # Top (Down)
                (0.0, 0.0, -1.0)  # Front (Backward)
            ]

        # store points and normals as arrays
        # converts to jax arrays with at least 2 dimensions
        self.points = jnp.atleast_2d(jnp.array(points))

        # safely ensure normals are unit vectors
        nms = jnp.atleast_2d(jnp.array(normals))

        self.normals = nms / jnp.linalg.norm(nms, axis=1, keepdims=True)

        # Here we broadcast friction
        if isinstance(frictions,float):
            num_walls = self.points.shape[0]
            self.frictions = jnp.broadcast_to(jnp.array(frictions), (num_walls,))
        else:
            self.frictions = frictions

        self.g_idx_list = g_idx_list
        self.f_idx = f_idx

    def create_state(self) -> BaseForceState:
        """Helper function to create empty state"""
        return BaseForceState()
        
    def apply_grid_moments(
        self, 
        mp_states, 
        grid_states, 
        f_states, 
        intr_caches, 
        couplings,
        dt, 
        time
    ):
        """
        Apply planar boundary conditions on background grids
        """
        for g_idx in self.g_idx_list:

            grid = grid_states[g_idx]

            # Get grid node coordinates compatible with C-Contiguous (Row-Major) layout
            indices = jnp.indices(grid.grid_size, dtype=jnp.float32)
    
            # Mesh (M_x, M_y, <_z, dim)
            coords = jnp.moveaxis(indices, 0, -1) * grid.cell_size + jnp.array(grid.origin)
            
            # Flatten to (num_nodes, dim)
            flat_coords = coords.reshape(-1, grid.dim)
            
            # Current grid velocities
            inv_mass = jnp.where(grid.mass_stack > 1e-14, 1.0 / grid.mass_stack, 0.0)
            vel = grid.moment_nt_stack  * inv_mass[:, None]
            
            def collide_all_walls(x, v_init):
            
                # The scan function runs for each wall
                def scan_fn(v_curr, params):
                    p, n, mu = params
                    v_next = self._collide_node(x, v_curr, p, n, mu)
                    return v_next, None

                # Scan over the walls (points, normals, frictions)
                v_final, _ = jax.lax.scan(
                    scan_fn, 
                    v_init, 
                    (self.points, self.normals, self.frictions)
                )
                return v_final

            # Here we vmap then scan over all points, normals and frictions
            # to cover whole domain
            new_vel = jax.vmap(collide_all_walls)(flat_coords, vel)
            
            # reconstruct Momentum
            new_mom = new_vel * grid.mass_stack[:, None]

            # update grid State
            new_grid = eqx.tree_at(lambda g: g.moment_nt_stack, grid, new_mom)
            
            # update global List index
            grid_states[g_idx] = new_grid
        
        return grid_states, f_states

    def _collide_node(self, x, v_node, point, normal, friction):
        """
        Calculates collision for a single node.
        """
        # check Penetration (signed distance)
        # Normal points OUT of the wall (into the domain)
        dist = jnp.dot(x - point, normal)
        
        # check velocity direction
        # v_n < 0 means moving INTO the wall
        v_n_mag = jnp.dot(v_node, normal)
        
        # check sdf
        # dist <= 0 implies the node is behind the plane
        is_colliding = (dist <= self.gap) & (v_n_mag < 0.0)

        def handle_collision(v_in):

            # decompose node velocities into normal and tangential
            v_n_vec = v_n_mag * normal
            v_t_vec = v_in - v_n_vec
            
            # Friction (coulomb)
            # We apply an impulse to stop normal motion (Delta_vn = -v_n_mag)
            # Max Tangential Impulse <= mu * Normal Impulse
            # => Delta_vt <= mu * |v_n_mag|

            vt_mag = jnp.linalg.norm(v_t_vec)
            vn_impulse = jnp.abs(v_n_mag) 
            
            # Calculate slip reduction
            # If friction is high, we subtract the full magnitude (Stick)
            # If friction is low, we subtract mu * vn
            reduction = friction * vn_impulse
            
            # New tangential magnitude (cannot go below zero)
            vt_new_mag = jnp.maximum(0.0, vt_mag - reduction)
            
            # Rescale tangent vector
            v_t_new = v_t_vec * (vt_new_mag / (vt_mag + 1e-12))
        
            return v_t_new 

        return jax.lax.cond(
            is_colliding,
            handle_collision, # Collision, returns new velosity
            lambda v: v, # No collision, return original
            v_node
        )