import numpy as np
import jax.numpy as jnp
import os
import sys

import pyvista as pv
from .simstate import SimState


try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

class VTKVisualizer:
    
    def __init__(self, 
                 output_dir="output_vtk", 
                 relative_dir = None
                 ):
        """
        Args:
            output_dir: Folder to save .vtp/.vti files.
            grid_topology: HydraxMPM grid topology.
            ppc: Particles per cell (used for radius calculation).

        """

        if relative_dir:
            # Get the directory where the script lives
            base_path = os.path.dirname(os.path.abspath(relative_dir))
            self.output_dir = os.path.join(base_path, output_dir)
        else:
            # Default behavior: Relative to where you ran the terminal command
            self.output_dir = output_dir

    def _save_vtk(self, mesh, label, filename):

        target_dir = os.path.join(self.output_dir, label)
        os.makedirs(target_dir, exist_ok=True)
        
        # Construct full file path
        full_path = os.path.join(target_dir, filename)
        mesh.save(full_path)
        return full_path
    
    def log_particles(
        self,
        mp_state,
        label="material_points",
        property_name="velocity_stack",
        scale_radius=1.0,
        time=None,
        step=None
    ):
        """
        Called every step. Writes one file per material per step.
        Example: output_vtk/material_0/step_00100.vtp
        """

        positions = np.array(mp_state.position_stack)
        num_points, dim = positions.shape
        volumes = np.array(mp_state.volume0_stack)

        if dim == 2:
            _point_radius = np.sqrt(volumes / np.pi)
        elif dim == 3:
            _point_radius = (3 * volumes / (4 * np.pi)) ** (1 / 3)

        _point_radius = _point_radius * scale_radius
        
        if property_name is not None:
            prop_data = getattr(mp_state, property_name, None)
        if positions.shape[1] == 2:
            zeros = np.zeros((positions.shape[0], 1))
            positions_3d = np.hstack([positions, zeros])
         
        else:
            positions_3d = positions

        # print(prop_data.mean())

        prop_data = np.array(prop_data)
        # print(prop_data.shape)

        if prop_data is not None:
            if len(prop_data.shape) > 1:
                if  prop_data.shape[1] == 2:
                    zeros = np.zeros((prop_data.shape[0], 1))
                    prop_data = np.hstack([prop_data, zeros])
            else:
                prop_data = prop_data

        cloud = pv.PolyData(positions_3d)

        cloud.point_data[property_name] = prop_data

        cloud.point_data["Radius"] = np.full(len(positions), _point_radius)
        
        if time is not None:
            time = float(time)
            cloud.field_data["TimeValue"] = np.array([time])


        step_val = int(step) if step is not None else 0
        path = self._save_vtk(cloud, label, f"step_{step_val:05d}.vtp")
        print(f"Material Points saved: {path}")


    def log_static_domain(self, origin, end, cell_size, label="domain"):
        """
        Creates a wireframe box representing the simulation domain.
        """
        origin = np.array(origin)

        end = np.array(end) + cell_size 
        

        # pyvista Box bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        if len(origin) == 2:
            bounds = (origin[0], end[0], origin[1], end[1], -0.01, 0.01)
        else:
            bounds = (origin[0], end[0], origin[1], end[1], origin[2], end[2])

        grid_box = pv.Box(bounds=bounds)
        
        path = self._save_vtk(grid_box, label, "bounds.vtp")
        print(f"Domain saved: {path}")
        
    def log_sdf(
            self,
            sdf_logic,
            sdf_state,
            start,
            end,
            resolution=100,
            label="boundary",
            step=None,
            time=None
        ):
            """
            Extracts the zero-level set of an SDF and saves it as a VTK mesh.
            """
            if not HAS_SKIMAGE:
                print("Install 'scikit-image' to visualize SDF boundaries in VTK.")
                return

            dim = len(start)
            start = np.array(start)
            end = np.array(end)
            
            # Setup sampling grid
            domain_size = end * 1.05 - start * 0.95
            step_size = np.max(domain_size) / resolution
            aspect_size = np.ceil(domain_size / step_size).astype(int)
            
            indices = np.indices(aspect_size)
            coords = np.moveaxis(indices, 0, -1) * step_size + start
            flat_coords = coords.reshape(-1, dim)

            # valuate SDF using JAX
            sdf_values = sdf_logic.get_signed_distance_stack(
                sdf_state, jnp.array(flat_coords)
            )
            sdf_grid = np.array(sdf_values).reshape(*aspect_size)

            mesh = None

            if dim == 2:
                # Extract Contours (2D)
                contours = measure.find_contours(sdf_grid, level=0.0)
                if len(contours) > 0:
                    all_points = []
                    all_lines = []
                    offset = 0
                    
                    for contour in contours:
                        # Convert to world coordinates
                        world_x = start[0] + contour[:, 0] * step_size
                        world_y = start[1] + contour[:, 1] * step_size
                        world_z = np.zeros_like(world_x)
                        pts = np.stack([world_x, world_y, world_z], axis=1)
                        
                        # Create connectivity: [num_pts, id0, id1, id2, ...]
                        num_pts = len(pts)
                        connectivity = np.hstack(([num_pts], np.arange(num_pts) + offset))
                        
                        all_points.append(pts)
                        all_lines.append(connectivity)
                        offset += num_pts
                    
                    # Combine all contours into one PolyData object
                    mesh = pv.PolyData(np.vstack(all_points), lines=np.hstack(all_lines))

            elif dim == 3:
                # Extract Surface (3D Marching Cubes)
                try:
                    verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)
                    world_verts = start + verts * step_size
                    # Faces connectivity: [3, v1, v2, v3, ...]
                    faces_vtk = np.hstack(np.c_[np.full(len(faces), 3), faces])
                    
                    mesh = pv.PolyData(world_verts, faces_vtk)
                    mesh.point_data["normals"] = normals
                except (RuntimeError, ValueError):
                    return

            if mesh is not None:
                if time is not None:
                    mesh.field_data["TimeValue"] = np.array([float(time)])

                suffix = f"step_{int(step):05d}" if step is not None else "static"
                # Standardized save via your _save_vtk method
                path = self._save_vtk(mesh, label, f"step_{int(step):05d}.vtp")