# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
"""
    Explaination:
        This module provides the logic and sdf_state for a Signed Distance Function (SDF) object.

    Features
        - Automatic differentiation for normal calculation. 
        Exact penetration normal without approximate mesh intersection algorithms (GJK/EPA), smooth boundaries.
        - Transformation between world and local SDF coordinates.
        - Vectorized batched interfaces

    Usage:
        - Generate material points 
        - Apply boundary conditions based on geometric shapes.

"""
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional, Self, Tuple

from jaxtyping import Float, Array

from ..utils.math_helpers import (
    rotation_2d_inv,
    quaternion_inv,
    quaternion_rotate,
    integrate_quaternion
)


class SDFObjectState(eqx.Module):
    """
    State for SDF Boundary Objects.

    Note the Velocities are defined relative to World/Body frame center of mass.

    Attributes:
        center_of_mass: Center of Mass position in world coordinates
        rotation: Scalar angle in 2D (radians), Quaternion [w, x, y, z] (Identity is [1, 0, 0, 0])
        velocity: Linear velocity of the SDF object
        angular_velocity: Scalar angular velocity in 2D (radians), quaternions [wx, wy, wz] in 3D.
    """
    center_of_mass: Float[Array, "dim"]
    rotation: float | Float[Array, "..."] | Float[Array, "4"]
    velocity: Float[Array, "dim"] 
    angular_velocity: float | Float[Array, "..."] | Float[Array, "3"]



class SDFObjectBase(eqx.Module):
    """
    Logic for Signed Distance Fields

    Note the SDF is assumed to be defined locally in cononical (object-local) space: (centered at (0,0,0), unrotated, unscaled)
    """
    def create_state(
        self,
        center_of_mass: Float[Array, "dim"],
        velocity: Optional[Float[Array, "dim"]] = None,
        angular_velocity: Optional[float | Float[Array, ""] | Float[Array, "3"]] = None,
        rotation: Optional[float | Float[Array, ""] | Float[Array, "4"]] = None,
    ) -> Self:
        """ Helper function to create default SDFObjectState """        
        dim = center_of_mass.shape[0]
        if velocity is None:
            velocity = jnp.zeros(dim)

        if dim == 2:
            if angular_velocity is None:
                angular_velocity = 0.0
            if rotation is None:
                rotation = 0.0
        else:
            if angular_velocity is None:
                angular_velocity = jnp.zeros(3)
            if rotation is None:
                rotation = jnp.array([1.0, 0.0, 0.0, 0.0])
        return SDFObjectState(
            center_of_mass=center_of_mass,
            velocity=velocity,
            angular_velocity=angular_velocity,
            rotation=rotation,
        )


    def transform_to_local(self, sdf_state: SDFObjectState, p_world: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """Transform world coordinates to local SDF coordinates using sdf_state."""
        # Relative coordinates to SDF
        p_local = p_world - sdf_state.center_of_mass

        # Apply rotation
        if p_world.shape[0] == 2:
            return rotation_2d_inv(sdf_state.rotation, p_local)
        else:
            return quaternion_rotate(quaternion_inv(sdf_state.rotation), p_local)

    
    def signed_distance_local(self, sdf_state: SDFObjectState,  p_local: Float[Array, "dim"]) -> Float[Array, ""]:
        """
        Get the signed distance for in a local coordinate system (Negative = Inside, Positive = Outside).
        
        Note these must be implemented by subclasses.
        """

        raise NotImplementedError


    def update_kinematics(
        self,
        sdf_state: SDFObjectState,
        dt,
    ):
        """Update the SDF object's state based on its velocity and angular velocity."""
        dim = sdf_state.center_of_mass.shape[0]

        new_pos = sdf_state.center_of_mass + sdf_state.velocity * dt

        if dim == 2:
            new_rot = sdf_state.rotation + sdf_state.angular_velocity * dt
        else:

            new_rot = integrate_quaternion(
                sdf_state.rotation, sdf_state.angular_velocity, dt
            )

        return eqx.tree_at(
            lambda s: (s.center_of_mass, s.rotation), sdf_state, (new_pos, new_rot)
        )
    
    def signed_distance(self, sdf_state: SDFObjectState,  pos_world: Float[Array, "dim"]) -> Float[Array, ""]:
        """
        Top level function to get signed distance in world coordinates. 
        """
        return self.signed_distance_local(sdf_state, self.transform_to_local(sdf_state, pos_world))
    

    def get_local_aabb(self, sdf_state:SDFObjectState) -> Tuple[Array, Array]:
        raise NotImplementedError
    
    def get_world_aabb(self, sdf_state) -> Tuple[Array, Array]:
        raise NotImplementedError
    
    def get_normal(self, sdf_state: SDFObjectState, pos_world: Float[Array, "dim"]) -> Float[Array, "dim"]:
        """
        Calculates Normal for a single point using Autograd by default.
        """
        grad_fn = jax.grad(self.signed_distance, argnums=1)
        
        normal = grad_fn(sdf_state, pos_world)
        
        # Safe normalization
        norm = jnp.linalg.norm(normal)
        return normal / (norm + 1e-12)
    def get_velocity(self, sdf_state: SDFObjectState, pos_world: Float[Array, "dim"],dt) -> Float[Array, "dim"]:
        """
        Calculates kinematic velocity for a single point.
        """

        p_local = pos_world - sdf_state.center_of_mass
        
        # v = v_lin + w x r
        # Handle 2D vs 3D cross product
        if p_local.shape[0] == 2:
            # 2D Cross product: omega is scalar, r is vector
            # [-w * ry, w * rx]
            cross = jnp.array([-sdf_state.angular_velocity * p_local[1], sdf_state.angular_velocity * p_local[0]])
        else:
            # 3D in WORLD frame
            cross = jnp.cross(sdf_state.angular_velocity, p_local)
            

        v_body = sdf_state.velocity + cross

        return v_body
    
    def get_surface_friction_local(self, sdf_state, p_local: Float[Array, "dim"]) -> Float[Array, ""] | float:
        """
        Returns the friction coefficient at this local point.
        Overridden by complex objects.
        """
        return self.friction if hasattr(self, 'friction') else 1.0

    def get_surface_friction(self, sdf_state, p_world: Float[Array, "dim"]) -> Float[Array, ""] | float:
        """Global wrapper."""
        return self.get_surface_friction_local(sdf_state, self.transform_to_local(sdf_state, p_world))


    def get_surface_friction_stack(self, sdf_state, p_world_stack):
        """Vectorized wrapper."""
        return jax.vmap(self.get_surface_friction, in_axes=(None, 0))(sdf_state, p_world_stack)
    

    def get_signed_distance_stack(self, sdf_state: SDFObjectState,  pos_world_stack: Float[Array, "n dim"]) -> Float[Array, "n"]:
        """Vectorized interface for SDF calculation."""
        return jax.vmap(self.signed_distance, in_axes=(None, 0))(sdf_state, pos_world_stack)

    def get_normal_stack(self, sdf_state: SDFObjectState, pos_world_stack: Float[Array, "n dim"]) -> Float[Array, "n dim"]:
        """Vectorized normals calculation."""
        return jax.vmap(self.get_normal, in_axes=(None, 0))(sdf_state, pos_world_stack)
    

    def get_velocity_stack(self, sdf_state: SDFObjectState, pos_world_stack: Float[Array, "n dim"],dt) -> Float[Array, "n dim"]:
        """Vectorized Velocity."""
        return jax.vmap(self.get_velocity, in_axes=(None, 0,None))(sdf_state, pos_world_stack, dt)
