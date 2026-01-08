import numpy as np
import jax.numpy as jnp
import os
import sys

import pyvista as pv
from .simstate import SimState




class VTKVisualizer:
    
    def __init__(self, 
                 output_dir="output_vtk", 
                 grid_topology=None, 
                 ppc=1,
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

        self.grid_topology = grid_topology
        self.ppc = ppc
        
        # 1. Setup Output Directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"📁 Created VTK output directory: {self.output_dir}")

        # 2. Pre-calculate Radius (as a data attribute for ParaView Glyphs)
        if grid_topology is not None:
            cell_size = grid_topology.cell_size 
            dim = len(grid_topology.origin)
            cell_vol = cell_size ** dim
            volume_per_particle = cell_vol / ppc
            
            if dim == 2:
                self._point_radius = np.sqrt(volume_per_particle / np.pi) 
            elif dim == 3:
                self._point_radius = (3 * volume_per_particle / (4 * np.pi)) ** (1 / 3)
        else:
            self._point_radius = 0.05

        # 3. Log the Static Domain (Grid) once at the start
        if self.grid_topology is not None:
            self._write_static_domain()

    def log_simulation(self, state: SimState):
        """
        Called every step. Writes one file per material per step.
        Example: output_vtk/material_0/step_00100.vtp
        """
        step = int(state.step)
        time = float(state.time)

        # Iterate through material points just like Rerun
        for i, mp in enumerate(state.material_points):
            self._write_particles(mp, material_id=i, step=step, time=time)

    def _write_particles(self, mp, material_id, step, time):
        # 1. Prepare Data (Convert JAX -> Numpy)
        positions = np.array(mp.position_stack)
        velocities = np.array(mp.velocity_stack)
        
        # VTK/ParaView requires 3D points. 
        # If 2D (N, 2), pad with z=0 to make (N, 3).
        if positions.shape[1] == 2:
            zeros = np.zeros((positions.shape[0], 1))
            positions_3d = np.hstack([positions, zeros])
            velocities_3d = np.hstack([velocities, zeros])
        else:
            positions_3d = positions
            velocities_3d = velocities

        # 2. Create PyVista Cloud
        # PolyData is best for unstructured particle clouds
        cloud = pv.PolyData(positions_3d)

        # 3. Add Attributes (Fields)
        # These appear in ParaView as "Point Data"
        cloud.point_data["Velocity"] = velocities_3d
        cloud.point_data["Velocity_Magnitude"] = np.linalg.norm(velocities_3d, axis=1)
        cloud.point_data["Radius"] = np.full(len(positions), self._point_radius)
        
        # Add Time as FieldData (Global)
        cloud.field_data["TimeValue"] = np.array([time])

        # 4. Save to Disk
        # Structure: output/material_X/step_YYYYY.vtp
        mat_dir = os.path.join(self.output_dir, f"material_{material_id}")
        if not os.path.exists(mat_dir):
            os.makedirs(mat_dir)

        filename = os.path.join(mat_dir, f"step_{step:05d}.vtp")
        print(filename)
        cloud.save(filename)

    def _write_static_domain(self):
        """Writes the grid definition as a .vti (Uniform Grid) file."""
        topo = self.grid_topology
        origin = np.array(topo.origin)
        end = np.array(topo.end)
        
        # Determine dimensions (number of nodes = cells + 1 usually, depending on definition)
        # Assuming topo.grid_size is number of cells
        dims = np.array(topo.grid_size) + 1
        spacing = (end - origin) / np.array(topo.grid_size)
        
        # Handle 2D -> 3D conversion for VTK
        if len(dims) == 2:
            dims = np.append(dims, 1)        # 1 layer thick in Z
            spacing = np.append(spacing, 0.1) # Arbitrary Z thickness
            origin_3d = np.append(origin, 0.0)
        else:
            origin_3d = origin

        # Create Uniform Grid
        grid = pv.ImageData()
        grid.dimensions = dims
        grid.spacing = spacing
        grid.origin = origin_3d

        # Save
        filename = os.path.join(self.output_dir, "domain_grid.vti")
        grid.save(filename)
