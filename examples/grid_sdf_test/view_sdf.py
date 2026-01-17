


import trimesh
from mesh_to_sdf import mesh_to_voxels
import numpy as np

if __name__ == "__main__":
    import hydraxmpm as hdx
    import os
    import jax.numpy as jnp
    dir_path = os.path.dirname(os.path.realpath(__file__))


    origin = jnp.array([-0.1, -0.1, -0.1])
    end = jnp.array([0.1, 0.1, 0.1])

    cell_size = ((end - origin)/64).min()

    # sdf = hdx.GridSDF(f"{dir_path}/stanford-bunny.npz", thickness=0.02, scale=1.0)  


    sdf = hdx.GridSDF(f"{dir_path}/cow.npz", thickness=0.0001, scale=0.01)  

    # print(sdf_cow.grid_min, sdf_cow.grid_size)

    smin, smax = sdf.get_local_bounds()
    print("Bunny SDF Local Bounds:", smin, smax)
    sdf_state = sdf.create_state(

        center_of_mass=jnp.array([0.,0.,0.])
    )

    vis = hdx.RerunVisualizer(
        origin = origin,
        end = end,
        cell_size=cell_size
        # origin=origin, end=end, cell_size=cell_size, ppc=ppc,
        #   mode="save"
        
        )
    
    vis.log_sdf_boundary(
    sdf_logic=sdf,
    sdf_state=sdf_state,
    resolution=200,
    )
