#  Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explanation:
    This module contains the real-time user interface visualizer using Rerun.
    see [https://rerun.io/](Rerun.io) for more information on the UI.



    Recommended to reset blueprint (left menu) when making significant changes to the simulation setup.

    Currently supports:
    - Material Points (colored by velocity magnitude)
    - Boundary
    - SDF output
    - 2D and 3D visualization


    Currently does not support visualization when using multi-GPU or distributed computing.

    Rerun can save a video using the following command:
    ```bash

    rerun record-session --app ./path/to/hydraxmpm_simulation_executable --output ./output_video.rrweb
    # or uv run rerun ... if using uv
    ```

"""
import rerun as rr
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt  # For colormaps

from jaxtyping import Array, Float
from .simstate import SimState

import os
import sys
import rerun.blueprint as rrb
try:
    from skimage import measure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


class RerunVisualizer:
    """
    Rerun Visualizer for real-time simulation visualization.

    Uses Rerun (https://rerun.io/) to visualize simulation states in real-time.

    Usage:

        Create visualizer before simulation loop:
        ```python
            vis = hdx.RerunVisualizer(
                origin=origin, end=end, cell_size=cell_size, ppc=ppc, app_name="HydraxMPM"
            )
        ```

        (Optional) Log SDF boundaries:
        ```python
            vis.log_sdf_boundary(
                sdf_logic=sdf_logic,
                sdf_state=sdf_state,
            )
        ```

        During simulation loop, log simulation state:
        ```python
            def log_simulation(sim_state):
                vis.log_simulation(sim_state)

            # usually called every N steps with jax.debug.callback
        ```

    Attributes:
        dim: Dimension of the simulation (2 or 3).
        cell_size: Size of each grid cell.
        origin: Origin coordinates of the simulation domain.
        end: End coordinates of the simulation domain.

    """
    ppc: int = 1
    dim: int = None
    cell_size: float = None
    origin: tuple = None
    end: tuple = None

   # Class-level flag to prevent opening multiple windows/servers
    _viewer_started = False 

    def __init__(
        self,
        app_name="HydraxMPM",
        recording_id= None,
        root_path="Sim_A",
        cmap=None,
        cell_size: float | Float[Array, "..."] = None,
        origin: Float[Array, "dim"] = None,
        end: Float[Array, "dim"] = None,
        ppc: int = 1,
        scale_radius: int = 1.0,
        mode="spawn"
    ):
        """Initialize the Rerun Visualizer.
        
        Args:
            app_name: Name of the Rerun application.
            cmap: Colormap for visualizing particle velocities.
            cell_size: Size of each grid cell.
            origin: Origin coordinates of the simulation domain.
            end: End coordinates of the simulation domain.
            ppc: Particles per cell (used for radius calculation).
            scale_radius: Scale factor for particle radius in visualization.

        """
        # Todo log grid nodes..

        # helps rerun find executable if not in PATH
        # e.g., when using virtual environments
        venv_bin = os.path.join(sys.prefix, "bin")
        if venv_bin not in os.environ["PATH"]:
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]



        # Store parameters
        self.origin = np.array(origin)
        self.end = np.array(end)

        self.cell_size = cell_size
        self.dim = len(origin)
        self.ppc = ppc

        # default colormap
        self.cmap = cmap if cmap is not None else plt.get_cmap("turbo")

        # Automatically calculate point radius based on ppc and cell size
        # Assume uniform cell size for simplicity
        if (
            (self.origin is not None)
            and (self.end is not None)
            and (self.cell_size is not None)
        ):
            cell_vol = self.cell_size**self.dim
            volume_per_particle = cell_vol / ppc
            if self.dim == 2:
                # 2D: Area = pi * r^2  ->  r = sqrt(Area / pi)
                self._point_radius = np.sqrt(volume_per_particle / np.pi)
            elif self.dim == 3:
                # 3D: Volume = 4/3 * pi * r^3  ->  r = cbrt(3*V / 4*pi)
                self._point_radius = (3 * volume_per_particle / (4 * np.pi)) ** (1 / 3)

            self._point_radius = self._point_radius * scale_radius
        else:
            self._point_radius = 0.05

        # Initialize the ID
        rr.init(app_name, recording_id=recording_id)
        self.root_path = root_path

        # Serve viewer once
        if not RerunVisualizer._viewer_started:
            if mode == "spawn":
                rr.spawn()
            elif mode == "serve":
                rr.serve(open_browser=False, server_addr="0.0.0.0")
            
            # Mark as started so we don't open a second window next time
            RerunVisualizer._viewer_started = True
            
        elif mode == "save":
            # Save mode is safe to run every time
            rr.save("simulation.rrd")
            
        if self.dim == 3:
            rr.log(f"{self.root_path}", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

        # Set up the camera for 3D visualization
        if (
            (self.origin is not None)
            and (self.end is not None)
            and (len(self.origin) == 3)
        ):

            max_dim = np.max(np.array(self.end))
            rr.log(
                f"{self.root_path}/camera",
                rr.Pinhole(focal_length=max_dim, width=max_dim * 2, height=max_dim * 2),
                static=True,
            )

        # Log static domain once at start
        if (
            (self.origin is not None)
            and (self.end is not None)
            and (self.cell_size is not None)
        ):
            self._log_static_domain()

        # if self.dim == 2:
        #     # Force a 2D View
        #     blueprint = rrb.Blueprint(
        #         rrb.Horizontal(
        #             rrb.Spatial2DView(origin=f"/{self.root_path}", name="2D Simulation")
        #         ),
        #         collapse_panels=True,
        #     )
        # else:
        #     # Force a 3D View
        #     blueprint = rrb.Blueprint(
        #         rrb.Horizontal(
        #             rrb.Spatial3DView(origin=f"/{self.root_path}", name="3D Simulation")
        #         ),
        #         collapse_panels=True,
        #     )
        
        # rr.send_blueprint(blueprint)

    def log_simulation(self, state: SimState):
        """
        Logs the current simulation state to Rerun.
        
        Sets the current step and time, and logs all material points.
        """
        
        current_step = int(state.step)
        current_time = float(state.time)
        rr.set_time("step", sequence=current_step)
        rr.set_time("sim_time", timestamp=current_time)
        
        # for i, mp in enumerate(state.world.material_points):
        #     self._log_particles(mp, f"{self.root_path}/material_{i}")

        # Add SDF logging here if needed?

    # def _log_particles(self, mp_state, path, cmap=None):
    #     """
    #     Logs the material points to Rerun. (internal use only)
    #     """
    #     # TODO add more flexibility on what to log (e.g., forces, stress, etc.)
    #     # Also colors 
    #     if cmap is None:
    #         cmap = plt.get_cmap("turbo")

    #     positions = np.array(mp_state.position_stack)

    #     # Color by velocity magnitude
    #     if mp_state.velocity_stack is None:
    #         colors = np.array([[200, 200, 200]] * positions.shape[0])  # Grey
    #     else:
    #         velocities = np.array(mp_state.velocity_stack)
    #         vel_mag = np.linalg.norm(velocities, axis=1)
            
    #         # Normalize (adjust based by physics)
    #         normalized_vel = np.clip(vel_mag / 2.0, 0, 1)
    #         colors = cmap(normalized_vel)

    #     # Rerun Logging
    #     if positions.shape[1] == 2:
    #         vis_pos = positions.copy()
    #         # Rerun flips Y axis for 2D visualizations
    #         vis_pos[:, 1] = -vis_pos[:, 1]
    #         rr.log(
    #             path,
    #             rr.Points2D(
    #                 vis_pos,
    #                 colors=colors,
    #                 radii=self._point_radius,  
    #             ),
    #         )
    #     else:
    #         rr.log(
    #             path,
    #             rr.Points3D(
    #                 positions,
    #                 colors=colors,
    #                 radii=self._point_radius, 
    #             ),
    #         )
    def _log_particles(self, 
                       mp_state,
                       label="material_points",
                       color=None,
                        property_name="velocity_stack",
                        cmap=None):
        """
        Logs the material points to Rerun. (internal use only)
        """
        # TODO add more flexibility on what to log (e.g., forces, stress, etc.)
        # Also colors 
        if cmap is None:
            cmap = plt.get_cmap("turbo")

        positions = np.array(mp_state.position_stack)
        num_points = positions.shape[0]

        if color is not None:
            c_arr = np.array(color)
            if c_arr.ndim == 1:
                colors = np.tile(c_arr, (num_points, 1))
            else:
                colors = c_arr

        else:
            prop_data = getattr(mp_state, property_name, None)

            if prop_data is None:
                # Fallback if property not found (Grey)
                print(f"Warning: Property '{property_name}' not found on state.")
                colors = np.array([[200, 200, 200]] * num_points)
            else:
                # Convert JAX array to Numpy
                values = np.array(prop_data)

                # 2. Reduce Vectors/Tensors to Scalars for visualization
                # If data is (N, dim) or (N, 3, 3), compute magnitude.
                # If data is (N,), use as is.
                if values.ndim > 1:
                    # Euclidean norm over all dimensions except the particle index (axis 0)
                    # e.g., for velocity (N, 3) -> axis 1
                    # e.g., for stress (N, 3, 3) -> axis (1, 2) (Frobenius norm equivalent)
                    reduce_axes = tuple(range(1, values.ndim))
                    scalar_field = np.linalg.norm(values, axis=reduce_axes)
                else:
                    scalar_field = values

                scalar_field = np.nan_to_num(scalar_field)

                v_min = np.min(scalar_field)
                v_max = np.max(scalar_field)
                
                # Avoid division by zero if field is constant
                if v_max - v_min < 1e-6:
                    normalized = np.zeros_like(scalar_field)
                else:
                    normalized = (scalar_field - v_min) / (v_max - v_min)

                colors = cmap(normalized)

                
        # Rerun Logging
        if positions.shape[1] == 2:
            vis_pos = positions.copy()
            # Rerun flips Y axis for 2D visualizations
            vis_pos[:, 1] = -vis_pos[:, 1]
            rr.log(
                f"{self.root_path}/{label}",
                rr.Points2D(
                    vis_pos,
                    colors=colors,
                    radii=self._point_radius,  
                ),
            )
        else:
            rr.log(
                f"{self.root_path}/{label}",
                rr.Points3D(
                    positions,
                    colors=colors,
                    radii=self._point_radius, 
                ),
            )


    def _log_static_domain(self):
        """Logs the bounding box and all grid nodes (timeless)."""

        self.end = self.end + np.ones(self.dim) * self.cell_size

        origin = self.origin.copy()
        end = self.end.copy()

        if self.dim == 2:
         
            # rerun flips the Y axis for 2D visualizations
            end[1] = -end[1]
            origin[1] = -origin[1]
            sizes = end - origin

            rr.log(
                f"{self.root_path}/domain/bounds",
                rr.Boxes2D(
                    mins=[origin],
                    sizes=[sizes],
                    labels=["Simulation Domain"],
                    colors=[[100, 100, 100]],  # Grey
                ),
                static=True,
            )

        elif self.dim == 3:
            sizes = self.end - self.origin
            rr.log(
                f"{self.root_path}/domain/bounds",
                rr.Boxes3D(
                    mins=[origin],
                    sizes=[sizes],
                    labels=["Simulation Domain"],
                    colors=[[100, 100, 100]],
                    fill_mode="wireframe",  # Wireframe box
                ),
                static=True,
            )
        # FUTURE
        # Log Background Nodes (plot faint dots)
        # rr.log(
        #     "world/domain/nodes",
        #     rr.Points2D(
        #         grid_coords,
        #         radii=0.002,
        #         # colors=[[50, 50, 50]] # Very dark grey
        #     ),
        #     static=True
        # )
        # rr.log(
        #     "world/domain/nodes",
        #     rr.Points3D(
        #         grid_coords,
        #         radii=0.002,
        #         colors=[[50, 50, 50]]
        #     ),
        #     static=True
        # )

    def log_sdf_boundary(self,
                        sdf_logic,
                        sdf_state,
                        resolution=100,
                        label="boundary", 
                        static=False,
                        start = None,
                        end = None
                        ):
        """
        Visualizes an SDF object by extracting the zero-level set (surface).
        Args:
            sdf_logic: sdf object logic.
            sdf_state: sdf object state (contains position/rotation).
            resolution: Grid resolution for sampling the SDF (higher = smoother but slower).
            static: Whether the SDF is static (timeless) or dynamic.
        """
        if not HAS_SKIMAGE:
            print("Install 'scikit-image' to visualize SDF boundaries.")
            return

        if start is None:
            start = self.origin

        if end is None:
            end = self.end
        

        # Here we generate a bunch of points in the domain
        domain_size = end*1.05 - start*0.95
        step_size = np.max(domain_size) / resolution

        aspect_size = np.ceil(domain_size / step_size).astype(int)
        indices = np.indices(aspect_size)
        coords = np.moveaxis(indices, 0, -1) * step_size + start
        flat_coords = coords.reshape(-1, self.dim)


        # Evaluate JAX SDF
        sdf_values = sdf_logic.get_signed_distance_stack(
            sdf_state, jnp.array(flat_coords)
        )

        # Reshape to grid
        sdf_grid = np.array(sdf_values).reshape(*aspect_size)

        if self.dim == 2:

            # extract Contours (lines where SDF=0)
            contours = measure.find_contours(sdf_grid, level=0.0)

            # collect all line strips
            all_strips = []
            for i, contour in enumerate(contours):

                # Get world coordinates
                world_x = self.origin[0] + contour[:, 0] * step_size
                world_y = self.origin[1] + contour[:, 1] * step_size

                # Apply Y-Flip for consistency with Rerun 2D and add to strips
                vis_y = -world_y

                points = np.stack([world_x, vis_y], axis=1)

                all_strips.append(points)
            rr.log(
                f"{self.root_path}/{label}",
                rr.LineStrips2D(
                    all_strips,
                    colors=[[255, 165, 0]],  # Orange
                    radii=0.01,
                    draw_order=100.0,
                ),
                static=static, # Can be static (timeless) if SDF does not move
            )

        elif self.dim == 3:
            # Same logic as above except using marching cubes to get a mesh, and aligning normals
            try:
                
                verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)

                # skimage returns (row/y, col/x, slice/z)
                world_x = self.origin[0] + verts[:, 0] * step_size
                world_y = self.origin[1] + verts[:, 1] * step_size
                world_z = self.origin[2] + verts[:, 2] * step_size
                world_verts = np.stack([world_x, world_y, world_z], axis=1)

                # Fix Normal permutation (skimage normals align with index axes Y, X, Z)
                # We need X, Y, Z
                world_normals = normals[:, [1, 0, 2]]

                rr.log(
                    f"{self.root_path}/{label}",
                    rr.Mesh3D(
                        vertex_positions=world_verts,
                        vertex_normals=world_normals,
                        triangle_indices=faces,
                        albedo_factor=[255, 165, 0, 128],  # Orange
                    ),
                    static=static,  # Can be static (timeless) if SDF does not move
                )
            except (RuntimeError, ValueError):
                # Happens if object is fully outside or fully inside (no surface found)
                pass


    def log_scalar(self, label: str, value: float | Array, group="metrics"):
        # 1. Convert JAX/Numpy to float
        if hasattr(value, "item"):
            val_float = float(value.item())
        else:
            val_float = float(value)

        path = f"{self.root_path}/{group}/{label}"

        # 2. Dynamic Dispatch based on installed version
        if hasattr(rr, "Scalar"):
            # Rerun 0.11+ (The Modern Standard)
            rr.log(path, rr.Scalar(val_float))
            
        elif hasattr(rr, "TimeSeriesScalar"):
            # Rerun 0.10 and older
            rr.log(path, rr.TimeSeriesScalar(val_float))
            
        else:
            # Fallback for very old or future-breaking versions
            # Some versions might allow direct logging, though deprecated
            try:
                rr.log_scalar(path, val_float)
            except AttributeError:
                print(f"Rerun Error: Cannot log scalar '{label}'. Update rerun-sdk.")

    def log_distribution(self, label: str, array: Array):
        data = np.array(array)
        rr.log(f"{self.root_path}/plots/{label}", rr.BarChart(data))