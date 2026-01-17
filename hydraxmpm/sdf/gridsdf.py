import equinox as eqx

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxtyping import Float, Array

from .sdfobject import SDFObjectBase,SDFObjectState

from ..utils.math_helpers import safe_norm

import jax
import numpy as np

class GridSDF(SDFObjectBase):
    """
    An SDF defined by a 3D Voxel Grid (lookup table).
    Used for imported STLs.
    """
    sdf_data: Float[Array, "nx ny nz"] 
    grid_min: Float[Array, "3"]
    grid_size: Float[Array, "3"] # Length of the box (max - min)

    thickness: float = eqx.field(static=True, default=1e-7)
    scale: float = eqx.field(static=True, default=1.0)

    def __init__(self, filename,thickness: float = 1e-7, scale: float = 1.0 ):
        # Load data using numpy (not jax.numpy) during init
        data = np.load(filename)
        self.sdf_data = jnp.array(data['sdf'])

        self.thickness = thickness
        self.scale = scale
        
        # Ensure data is 3D
        if self.sdf_data.ndim != 3:
            raise ValueError("SDF Data must be 3D")
            
        self.grid_min = jnp.array(data['min'])
        self.grid_size = jnp.array(data['max']) - self.grid_min

    def get_local_bounds(self):
        """
        Returns the min and max corners in Local Space, 
        accounting for Scale and Thickness.
        """
        # 1. Scale the raw grid bounds
        s_min = self.grid_min * self.scale
        s_max = (self.grid_min + self.grid_size) * self.scale
        
        # 2. Expand by thickness 
        # (Thickness makes the object "fatter", pushing bounds out)
        s_min = s_min - self.thickness
        s_max = s_max + self.thickness
        
        return s_min, s_max
    
    def signed_distance_local(self, state, p_local):
        """
        Maps local position to grid index and interpolates.
        """

        p_unscaled = p_local / self.scale

        # 1. Normalize position to [0, 1] inside the bounding box
        # p_norm = (p - min) / size
        p_norm = (p_unscaled - self.grid_min) / (self.grid_size + 1e-12)
        
        # 2. Convert to Grid Indices (0 to Resolution-1)
        # resolution = shape - 1
        res = jnp.array(self.sdf_data.shape) - 1.0
        indices = p_norm * res
        
        # 3. Differentiable Interpolation
        # jax.scipy.ndimage.map_coordinates works exactly like 
        # texture sampling in a shader.
        # order=1 (Linear) is fast and differentiable.
        # order=3 (Cubic) gives smoother normals but costs more.
        dist = jax.scipy.ndimage.map_coordinates(
            self.sdf_data, 
            indices.reshape(3, 1), 
            order=1, 
            mode='nearest' # Clamp to edge
        )
        
        return dist[0]*self.scale - self.thickness