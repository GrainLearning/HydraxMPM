import trimesh
from mesh_to_sdf import mesh_to_voxels
import numpy as np


def convert_stl_to_sdf_grid(filename, save_path, resolution=128):
    # 1. Load Mesh
    mesh = trimesh.load(filename)
    mesh.apply_translation(-mesh.centroid)

    # 2. CALCULATE CUBIC BOUNDS (Keep this!)
    max_extent = np.max(mesh.extents)
    box_size = max_extent * 1.1
    half_size = box_size / 2.0

    # Define the target Cubic Bounds
    # (We save THESE variables later, do not overwrite them!)
    target_min = np.array([-half_size, -half_size, -half_size])
    target_max = np.array([half_size, half_size, half_size])

    # 3. FORCE VOXELIZER TO SEE A CUBE
    # The mesh_to_voxels function tries to be smart and crop empty space.
    # To prevent it from cropping our nice cube back into a rectangle,
    # we add tiny dummy vertices at the corners of our cube.
    
    corners = np.array([
        target_min,
        target_max,
        [-half_size, -half_size, half_size],
        [half_size, -half_size, -half_size],
        # ... just min and max is usually enough to force the bounding box
    ])
    
    # Create a point cloud of corners and combine with mesh vertices
    # (We don't need faces, just extent)
    # Actually, mesh_to_voxels might ignore loose vertices.
    # Let's Scale instead, which is safer.
    
    scale_factor = 2.0 / box_size 
    mesh.apply_scale(scale_factor)
    
    # 4. Compute SDF
    # Now the mesh fits inside [-1, 1].
    # mesh_to_voxels will scan the unit cube because the mesh now occupies it.
    voxels = mesh_to_voxels(
        mesh,
        voxel_resolution=resolution,
        pad=True, 
        sign_method="depth",
        surface_point_method="scan",
    )

    # 5. Fix Distances
    # The voxels are in "Unit Scaled Space" (distances are roughly 0.0 to 1.0)
    # We need to scale them back to "World Space" distances.
    voxels = voxels / scale_factor

    # 6. SAVE
    # BUG FIX: Use 'target_min' and 'target_max', NOT 'mesh.bounds'
    np.savez(save_path, sdf=voxels, min=target_min, max=target_max)
    
    print(f"Saved {save_path}.")
    print(f"Grid Shape: {voxels.shape}")
    # print(f"Physical Bounds: {target_min} to {target_max} (Perfect Cube)")


if __name__ == "__main__":
    import os

    dir_path = os.path.dirname(os.path.realpath(__file__))

    # convert_stl_to_sdf_grid(
    #     f"{dir_path}/stanford-bunny.obj",
    #     f"{dir_path}/stanford-bunny.npz",
    #     resolution=128,
    # )

    convert_stl_to_sdf_grid(
        f"{dir_path}/cow.obj",
        f"{dir_path}/cow.npz",
        resolution=128,
    )
