
# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
"""
Explanation:
    This module contains various Signed Distance Function (SDF) objects for defining boundaries and obstacles.
    Each SDF object implements the signed_distance_local method to compute the SDF in local coordinates.

    References:
    - Inigo Quilez. "Distance Functions." https://iquilezles.org/

"""
import equinox as eqx

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxtyping import Float, Array

from .sdfobject import SDFObjectBase,SDFObjectState

from ..utils.math_helpers import safe_norm

class BoxSDF(SDFObjectBase):
    """
    Axis-aligned Box box centered at state.center_of_mass.
    size is lengths in each dimension.

    Domain: finite
    Shape: Box (2D) or Rectangular Prism (3D)
    Usage: Obstacles, Containers, boundaries, walls, blocks

    """
    half_size: Float[Array, "dim"] # Static shape parameter

    def __init__(self, size):
        size = jnp.asarray(size)
        self.half_size = size / 2.0

    def signed_distance_local(self, state: SDFObjectState, pos_local: Float[Array, "dim"]) -> Float[Array, ""]:
        d = jnp.abs(pos_local) - self.half_size
        d_vec = jnp.maximum(d, 0.0)
        outside = safe_norm(d_vec)
        inside = jnp.minimum(jnp.max(d), 0.0)
        
        return outside + inside
    
class SphereSDF(SDFObjectBase):
    """
    A Sphere (3D) or Circle (2D) centered at `state.center_of_mass`.

    Shape: Sphere (3D) or Circle (2D)
    Domain: Finite, 2D/3D
    Usage: Obstacles, Boundaries, Rigid Bodies

    """
    radius: float

    def __init__(self, radius):
        self.radius = radius

    def signed_distance_local(self, state: SDFObjectState, p_local: Float[Array, "dim"]) -> Float[Array, ""]:        
        # length(p) - r
        return jnp.linalg.norm(p_local) - self.radius
    
    def signed_distance(self, state: SDFObjectState, pos_world: Float[Array, "dim"]) -> Float[Array, ""]:
        # override to avoid extra transform step
        p_local = pos_world - state.center_of_mass
        return self.signed_distance_local(state, p_local)


    
class PlaneSDF(SDFObjectBase):
    """
    An infinite plane defined by a normal vector.
    The plane passes through the `state.center_of_mass`.

    Shape: Flat Sheet
    Domain: Infinite (divides world in two), 2D/3D
    Usage: Floor, Walls, Ceilings
    """
    normal: Float[Array, "dim"]

    def __init__(self, normal):
        n = jnp.asarray(normal)
        self.normal = n / (jnp.linalg.norm(n) + 1e-12)

    def signed_distance_local(self, state: SDFObjectState, pos_local: Float[Array, "dim"]) -> Float[Array, ""]:
        return jnp.dot(pos_local, self.normal)


class CapsuleSDF(SDFObjectBase):
    """
    A capsule defined by height (length between sphere centers) and radius.

    Shape: A vertical capsule (pill)
    Domain: Finite, aligned with Y axis, 2D/3D
    Usage: Obstacles, Boundaries, Rigid Bodies
    """
    radius: float
    half_height: float

    def __init__(self, radius, height):
        self.radius = radius
        self.half_height = height / 2.0

    def signed_distance_local(self, state: SDFObjectState, p_local: Float[Array, "dim"]) -> Float[Array, ""]:

        # clamp the point's Y coordinate to the line segment [-h, h]
        p_y_clamped = jnp.clip(p_local[1], -self.half_height, self.half_height)
        
        # Create the closest point on the line segment
        closest_on_segment = jnp.zeros_like(p_local).at[1].set(p_y_clamped)
        
        # Distance is dist(p, segment) - radius
        return jnp.linalg.norm(p_local - closest_on_segment) - self.radius


class HollowCylinderSDF(SDFObjectBase):
    """
    A hollow cylinder (pipe) defined by inner and outer radius, and height.

    
    3D: A vertical pipe (Y-axis) with wall thickness.
    2D (Default): Two vertical walls (Cross-section of pipe from side).
    2D (Ring): An Annulus/Washer (Cross-section of pipe from top).
    
    """
    height: float
    r_outer: float
    r_inner: float
    is_2d_ring: bool = eqx.field(static=True)

    def __init__(self, height, outer_radius, inner_radius, is_2d_ring=True):
        self.height = height
        self.r_outer = outer_radius
        self.r_inner = inner_radius
        self.is_2d_ring = is_2d_ring

    def signed_distance_local(self, state, pos_local):
        if pos_local.shape[0] == 3:
            # 2D distance from Y axis
            r = jnp.linalg.norm(pos_local[jnp.array([0, 2])])
            d_y = pos_local[1]
        else:
            if self.is_2d_ring:
                # Annulus / Ring (Top Down View)
                # Radial distance is magnitude of vector (x,y)
                r_xz = jnp.abs(pos_local[0])
                r = jnp.linalg.norm(pos_local)

                # Height is irrelevant/infinite for a 2D ring shape
                d_y = -1e9
            else:
                # Parallel Walls (Side View)
                # Radial distance is just X (distance from axis)
                r = jnp.abs(pos_local[0])
                # Height bounds apply to Y
                d_y = jnp.abs(pos_local[1]) - (self.height / 2.0)

            y = pos_local[1]


        # Distance to the solid part of the pipe (between r_inner and r_outer)
        d_radial = jnp.maximum(r - self.r_outer, self.r_inner - r)


        d_vec = jnp.array([d_radial, d_y])
        
        d_vec_clamped = jnp.maximum(d_vec, 0.0)
        outside = safe_norm(d_vec_clamped)
        
        inside = jnp.minimum(jnp.max(d_vec), 0.0)
        
        return outside + inside

class CylinderSDF(SDFObjectBase):
    """
   
    Capped Cylinder  SDF

    Shape:   A Vertical Cylinder with flat caps (Y-axis)
    Domain: Finite, aligned with Y axis, 3D
    Usage: Obstacles, Boundaries, Rigid Bodies

    """
    radius: float
    half_height: float

    def __init__(self, radius, height):
        self.radius = radius
        self.half_height = height / 2.0

    def signed_distance_local(self, state: SDFObjectState, pos_local: Float[Array, "dim"]) -> Float[Array, ""]:
        
        # Radial distance in XZ plane
        d_xz = safe_norm(pos_local[jnp.array([0, 2])]) - self.radius
        
        # Vertical distance
        d_y = jnp.abs(pos_local[1]) - self.half_height
        
        # Combine interior/exterior distances (Logic similar to Box)
        # Vector of distances to the side surface and the cap surface
        d_vec = jnp.array([d_xz, d_y])
        
        outside = safe_norm(jnp.maximum(d_vec, 0.0))
        inside = jnp.minimum(jnp.max(d_vec), 0.0)
        
        return outside + inside



class TorusSDF(SDFObjectBase):
    """
    A Torus (Donut) lying flat on the XZ plane.

    major_r: Distance from center to the middle of the tube.
    minor_r: Radius of the tube itself.

    Shape: Torus (Donut)
    Domain: finite, aligned with XZ plane, 3D only
    Usage: Obstacles, Boundaries, Rigid Bodies

    """
    major_r: float
    minor_r: float

    def __init__(self, major_r, minor_r):
        self.major_r = major_r
        self.minor_r = minor_r

    def signed_distance_local(self, state: SDFObjectState, pos_local: Float[Array, "dim"]) -> Float[Array, ""]:

        # Project the position onto the XZ plane to find horizontal distance
        q_x =safe_norm(pos_local[jnp.array([0, 2])]) - self.major_r
        q_y = pos_local[1]
        
        q = jnp.array([q_x, q_y])
        
        # Distance from the circle is the norm minus tube radius
        return jnp.linalg.norm(q) - self.minor_r
    

class StarSDF(SDFObjectBase):
    """
    A 2D Star shape with N points.
    
    Parameters:
        points (n): Number of arms (e.g., 5).
        outer_radius (r): Distance from center to tip of an arm.
        inner_radius (r_in): Distance from center to the valley between arms.
    """
    points: float = eqx.field(static=True)
    outer_radius: Float[Array, ""] 
    inner_radius: Float[Array, ""]

    def __init__(self, points: int = 5, outer_radius: float = 1.0, inner_radius: float = 0.5):
        self.points = float(points)
        self.outer_radius = jnp.asarray(outer_radius)
        self.inner_radius = jnp.asarray(inner_radius)

    def signed_distance_local(self, state: SDFObjectState, p: Float[Array, "dim"]) -> Float[Array, ""]:
        # 2D Star SDF based on Inigo Quilez method
        # https://iquilezles.org/articles/distfunctions2d/

        an = jnp.pi / self.points
        
        en = jnp.array([jnp.cos(an), jnp.sin(an)]) # needed?
        
        # Symmetry Folding (Polar)
        angle = jnp.arctan2(p[0], p[1]) # Align 0 with Y-axis (Standard star orientation)
        
  
        # e.g. if angle is 37 deg and sector is 72 deg, index = 1
        sector_idx = jnp.round(angle / (2.0 * an))

        # We treat sector_idx as a continuous variable for math, but it acts as an integer
        sector_angle = sector_idx * (2.0 * an)
        
        # Rotation Matrix (2D) to bring p into the wedge
        c, s = jnp.cos(sector_angle), jnp.sin(sector_angle)
        
        # p_wedge is "p" rotated so it aligns with the Y-axis sector
        p_wedge = jnp.array([
            p[0] * c - p[1] * s,
            p[0] * s + p[1] * c
        ])
        
        # 3. Symmetry Folding (Mirror)
        p_wedge = p_wedge.at[0].set(jnp.abs(p_wedge[0]))
        
        # distance to Line Segment
        # The edge of the star connects:
        #   v1: The valley (inner_radius projected by angle 'an')
        #   v2: The tip (0, outer_radius)
        v1 = jnp.array([0.0, self.outer_radius])
        
        # Vertex 2 (Valley of the star, rotated by 'an')
        v2 = jnp.array([self.inner_radius * jnp.sin(an), self.inner_radius * jnp.cos(an)])
        
        # Vector along the edge
        edge = v2 - v1
        # Vector from Tip to Point
        point_to_v1 = p_wedge - v1
        
        # Project point onto the edge (clamped to segment [0, 1])
        # h = clamp( dot(point_to_v1, edge) / dot(edge, edge), 0.0, 1.0 )
        h = jnp.clip(jnp.dot(point_to_v1, edge) / jnp.dot(edge, edge), 0.0, 1.0)
        
        # Closest point on the segment
        closest = v1 + h * edge
        
        # Distance vector
        dist_vec = p_wedge - closest
        dist = jnp.linalg.norm(dist_vec)
        
        # sign Correction (Inside vs Outside)
        cross = edge[0] * (p_wedge[1] - v1[1]) - edge[1] * (p_wedge[0] - v1[0])
        
        # If cross > 0, we are "above/right" (outside). If < 0, inside.
        return dist * jnp.sign(cross)
