import hydraxmpm as hdx

import jax.numpy as jnp

import equinox as eqx

class HorizontalSemiCylinderSDF(hdx.SDFObjectBase):
    length: float
    radius: float
    level_offset: float

    def __init__(self, length, radius, level_offset=-1.0):
        self.length = length
        self.radius = radius
        self.level_offset = level_offset

    def signed_distance_local(self, state, p):
        # 1. CYLINDER LOGIC (Along X-Axis)
        # Distance from the center line (X-axis) is length of vector (y, z)
        # We use existing math: length(yz) - radius
        d_yz = jnp.linalg.norm(p[jnp.array([1, 2])]) - self.radius
        
        # 2. LENGTH LOGIC (Along X-Axis)
        d_x = jnp.abs(p[0]) - (self.length / 2.0)
        # self.level_offset = level_offset
        
        # 3. HALF-CYCLE LOGIC (Cut off the top)
        # We want particles only in the bottom half (y < 0).
        # So if y > 0, we return positive distance (outside).
        # d_plane = p[1] 
        d_plane = p[1] - self.level_offset

        # Intersection: Max of (Cylinder, Length_Bounds, Top_Cut)
        # We use a smooth intersection or hard max. Hard max is fine for generation.
        
        # Combine cylinder tube and length caps
        d_tube = jnp.maximum(d_yz, d_x)
        
        # Combine tube and the "half" plane cut
        return jnp.maximum(d_tube, d_plane)
    


class MixerCylinderSDF(hdx.SDFObjectBase):
    """
    A hollow cylinder with internal 'lifter bars' (bevels) for mixing.
    """
    height: float
    r_outer: float
    r_inner: float
    
    # Bevel/Mixer dimensions
    bevel_height: float  # How far it sticks out into the center
    bevel_width: float   # How thick the bar is
    
    num_mixers: int = eqx.field(static=True) # Static for JIT compilation

    def __init__(
        self, 
        height, 
        outer_radius, 
        inner_radius, 
        num_mixers=4, 
        bevel_height=0.5, 
        bevel_width=0.2
    ):
        self.height = height
        self.r_outer = outer_radius
        self.r_inner = inner_radius
        self.num_mixers = num_mixers
        self.bevel_height = bevel_height
        self.bevel_width = bevel_width

    def signed_distance_local(self, state, p):
        # --- 1. HOLLOW CYLINDER LOGIC (Same as before) ---
        # Vertical Cylinder (Axis Y)
        r = hdx.safe_norm(p[jnp.array([0, 2])]) 
        
        # Distance to the pipe walls
        d_radial = jnp.maximum(r - self.r_outer, self.r_inner - r)
        # Distance to the top/bottom caps
        d_y = jnp.abs(p[1]) - (self.height / 2.0)

        # Combine for Cylinder
        d_vec_cyl = jnp.array([d_radial, d_y])
        outside_cyl = hdx.safe_norm(jnp.maximum(d_vec_cyl, 0.0))
        inside_cyl = jnp.minimum(jnp.max(d_vec_cyl), 0.0)
        dist_cylinder = outside_cyl + inside_cyl

        # --- 2. BEVEL/MIXER LOGIC (Polar Repetition) ---
        
        # We work in the XZ plane (cross section)
        x, z = p[0], p[2]
        
        # Get the angle of the current point
        angle = jnp.arctan2(z, x)
        
        # Calculate sector size (e.g., 90 degrees for 4 mixers)
        sector_step = 2 * jnp.pi / self.num_mixers
        
        # Find which sector we are in (round to nearest index)
        sector_idx = jnp.round(angle / sector_step)
        
        # Calculate the angle to rotate this sector to align with the X-axis
        rot_angle = sector_idx * sector_step
        
        # Rotate point (x,z) by -rot_angle so it aligns with X-axis
        # This creates the "Folded" space. We only define ONE box on the X-axis now.
        c = jnp.cos(rot_angle)
        s = jnp.sin(rot_angle)
        
        x_rot = x * c + z * s
        z_rot = -x * s + z * c
        
        # Define the Box (Lifter Bar)
        # It is located on the inner wall (x = r_inner)
        # It sticks inward, so center x is roughly r_inner
        
        # Box Center in the folded space
        # We shift it slightly so it protrudes cleanly from the wall
        box_center_x = self.r_inner
        
        # Calculate distance to Box (Standard Box SDF)
        # dist = abs(p - center) - size
        d_box_x = jnp.abs(x_rot - box_center_x) - (self.bevel_height / 2.0)
        d_box_z = jnp.abs(z_rot) - (self.bevel_width / 2.0)
        d_box_y = jnp.abs(p[1]) - (self.height / 2.0) # Same height as cylinder

        # Box SDF logic
        d_vec_box = jnp.array([d_box_x, d_box_y, d_box_z])
        outside_box = hdx.safe_norm(jnp.maximum(d_vec_box, 0.0))
        inside_box = jnp.minimum(jnp.max(d_vec_box), 0.0)
        dist_box = outside_box + inside_box

        # --- 3. UNION ---
        # Union of solids = Minimum of signed distances
        # We want the Union of the Pipe Walls AND the Mixer Bars.
        # Since both return negative values when inside, min() works perfectly.
        
        return jnp.minimum(dist_cylinder, dist_box)