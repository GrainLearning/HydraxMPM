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
from uuid import uuid4
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
        recording_id=None,
        root_path="Sim_A",
        mode="spawn",
        is_3d=False,
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
        if recording_id is None:
            recording_id = str(uuid4())

        # helps rerun find executable if not in PATH
        # e.g., when using virtual environments
        venv_bin = os.path.join(sys.prefix, "bin")
        if venv_bin not in os.environ["PATH"]:
            os.environ["PATH"] = venv_bin + os.pathsep + os.environ["PATH"]

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
            rr.save("simulation.rrd")

        if is_3d:
            view = rrb.Spatial3DView(
                name=f"Simulation view", origin=self.root_path + "/"
            )
            blueprint = rrb.Grid(view)

            rr.log(f"{self.root_path}", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
            rr.send_blueprint(blueprint, make_active=True)

    def log_time(self, current_step: int, current_time: float):

        rr.set_time("step", sequence=current_step)
        rr.set_time("sim_time", timestamp=current_time)

    def log_material_points(
        self,
        mp_state,
        label="material_points",
        color=None,
        property_name="velocity_stack",
        cmap=None,
        # recording_stream=None,
        ppc=1,
        scale_radius=1.0,
        v_min=None,
        v_max=None
    ):
        """
        Logs the material points to Rerun. (internal use only)
        """
        # TODO add more flexibility on what to log (e.g., forces, stress, etc.)
        # Also colors
        if cmap is None:
            cmap = plt.get_cmap("turbo")

        positions = np.array(mp_state.position_stack)
        num_points, dim = positions.shape
        volumes = np.array(mp_state.volume0_stack)

        if dim == 2:
            _point_radius = np.sqrt(volumes / np.pi)
        elif dim == 3:
            _point_radius = (3 * volumes / (4 * np.pi)) ** (1 / 3)

        _point_radius = _point_radius * scale_radius

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

                if v_min is None:
                    v_min = np.min(scalar_field)
                if v_max is None:
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
                    radii=_point_radius,
                ),
                # recording=recording_stream,
            )
        else:
            rr.log(
                f"{self.root_path}/{label}",
                rr.Points3D(
                    positions,
                    colors=colors,
                    radii=_point_radius,
                ),
            )

    def log_static_domain(self, origin=None, end=None, cell_size=None, label="domain"):
        """Logs the bounding box and all grid nodes (timeless)."""
        dim = len(origin)
        origin = np.array(origin)
        end = np.array(end)
        end = end + np.ones(dim) * cell_size

        if dim == 2:

            # rerun flips the Y axis for 2D visualizations
            end[1] = -end[1]
            origin[1] = -origin[1]
            sizes = end - origin

            rr.log(
                f"{self.root_path}/{label}",
                rr.Boxes2D(
                    mins=[origin],
                    sizes=[sizes],
                    colors=[[100, 100, 100]],  # Grey
                ),
                static=True,
            )

        elif dim == 3:
            sizes = end - origin
            rr.log(
                f"{self.root_path}/{label}",
                rr.Boxes3D(
                    mins=[origin],
                    sizes=[sizes],

                    colors=[[100, 100, 100]],
                    fill_mode="wireframe",  # Wireframe box
                ),
                static=True,
            )

    def log_sdf(
        self,
        sdf_logic,
        sdf_state,
        resolution=100,
        label="boundary",
        static=False,
        start=None,
        end=None,
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
        dim = len(start)

        end = np.array(end)
        start = np.array(start)
        # Here we generate a bunch of points in the domain
        domain_size = end * 1.05 - start * 0.95
        step_size = np.max(domain_size) / resolution

        aspect_size = np.ceil(domain_size / step_size).astype(int)
        indices = np.indices(aspect_size)
        coords = np.moveaxis(indices, 0, -1) * step_size + start
        flat_coords = coords.reshape(-1, dim)

        # Evaluate JAX SDF
        sdf_values = sdf_logic.get_signed_distance_stack(
            sdf_state, jnp.array(flat_coords)
        )

        # Reshape to grid
        sdf_grid = np.array(sdf_values).reshape(*aspect_size)

        if dim == 2:

            # extract Contours (lines where SDF=0)
            contours = measure.find_contours(sdf_grid, level=0.0)

            # collect all line strips
            all_strips = []
            # print(f"Extracted {len(contours)} contour(s) for SDF boundary visualization.")
            for i, contour in enumerate(contours):

                # Get world coordinates
                world_x = start[0] + contour[:, 0] * step_size
                world_y = start[1] + contour[:, 1] * step_size

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
                static=static,  # Can be static (timeless) if SDF does not move
            )

        elif dim == 3:

            try:

                verts, faces, normals, _ = measure.marching_cubes(sdf_grid, level=0.0)

                # skimage returns (row/y, col/x, slice/z)
                world_x = start[0] + verts[:, 0] * step_size
                world_y = start[1] + verts[:, 1] * step_size
                world_z = start[2] + verts[:, 2] * step_size
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

