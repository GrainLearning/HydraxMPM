from typing import Any, Dict, List

import numpy as np
import pyvista as pv


def points_to_3D(points, dim=2):
    """Convert 2D points to 3D points."""
    if dim == 3:
        return np.array(points)
    return np.pad(points, [(0, 0), (0, 1)], mode="constant")


def add_cloud_mesh(
    positions: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str,
    params: Dict[str, Any],
    plotter: pv.Plotter,
):
    """Add point cloud to plotter."""
    num_points, dim = positions[0].shape

    cloud = pv.PolyData(points_to_3D(positions[0], dim))

    if scalars is not None:
        cloud.point_data[scalar_name] = scalars[0]

    if params is None:
        params = {}

    params.setdefault("style", "points")
    params.setdefault("cmap", "viridis")
    params.setdefault("show_edges", True)
    params.setdefault("scalar_bar_args", dict(position_x=0.2, position_y=0.01))
    params.setdefault("scalars", scalar_name)
    params.setdefault("clim", [-0.1, 0.1])

    plotter.add_mesh(cloud, **params)
    return plotter, cloud


def add_box_mesh(
    origin: np.ndarray, end: np.ndarray, params: Dict[str, Any], plotter: pv.Plotter
):
    """Add box mesh to plotter."""
    if len(origin) == 2:  # dim==2
        origin = np.pad(origin, [(0, 1)], mode="constant")
        end = np.pad(end, [(0, 1)], mode="constant")

    box = pv.Box(bounds=np.array(list(zip(origin, end))).flatten())

    if params is None:
        params = {}

    params.setdefault("color", "white")
    params.setdefault("style", "wireframe")

    plotter.add_mesh(box, **params)
    return plotter, box


def set_camera(params: Dict[str, Any], dim: int, plotter: pv.Plotter):
    """Set camera position and zoom for 2D or 3D plots."""
    if params is None:
        params = {}
    if dim == 2:
        params.setdefault("camera_position", "xy")
        params.setdefault("azimuth", 0)
        params.setdefault("elevation", 0)
        params.setdefault("zoom", 1.1)
    else:
        params.setdefault("camera_position", "xz")
        params.setdefault("azimuth", 45)
        params.setdefault("elevation", 30)
        params.setdefault("zoom", 0.9)

    plotter.camera_position = params["camera_position"]
    plotter.camera.azimuth = params["azimuth"]
    plotter.camera.elevation = params["elevation"]
    plotter.camera.zoom(params["zoom"])
    return plotter


def plot_simple(
    origin: np.ndarray,
    end: np.ndarray,
    positions_stack: List[np.ndarray],
    scalars: List[np.ndarray] = None,
    scalars_name: str = None,
    rigid_positions_stack: List[np.ndarray] = None,
    box_plot_params: Dict[str, Any] = None,
    particles_plot_params: Dict[str, Any] = None,
    rigid_plot_params: Dict[str, Any] = None,
    theme: str = "default",
    output_file: str = "output.gif",
):
    """Pyvista wrapper to plot MPM simulation.

    Args:
        origin (np.ndarray): Domain origin
        end (np.ndarray): Domain end
        positions_stack (List[np.ndarray]): List of particle positions
        scalars (List[np.ndarray], optional): List of scalars. Defaults to None.
        scalars_name (str, optional): Name of scalar. Defaults to None.
        rigid_positions_stack (List[np.ndarray], optional):
            List of rigid particle positions. Defaults to None.
        box_plot_params (Dict[str, Any], optional):
            Plot params of domain boundary. Defaults to None.
        particles_plot_params (Dict[str, Any], optional):
            Plot params of particles. Defaults to None.
        rigid_plot_params (Dict[str, Any], optional):
            Plot params of rigid particles. Defaults to None.
        theme (str, optional): Pyvista theme to plot. Defaults to "default".
        output_file (str, optional): Output file name. Defaults to "output.gif".
    """
    pv.set_plot_theme(theme)

    pl = pv.Plotter(notebook=False, off_screen=True)
    pl, box = add_box_mesh(origin, end, box_plot_params, pl)

    pl, particles_cloud = add_cloud_mesh(
        positions_stack, scalars, scalars_name, particles_plot_params, pl
    )

    if rigid_positions_stack is not None:
        pl, rigid_cloud = add_cloud_mesh(
            rigid_positions_stack, None, None, rigid_plot_params, pl
        )

    pl = set_camera(particles_plot_params, positions_stack[0].shape[1], pl)

    pl.open_gif(output_file)

    for iter in range(len(positions_stack)):
        positions = positions_stack[iter]

        num_points, dim = positions.shape

        particles_cloud.points = points_to_3D(positions, dim)

        if scalars is not None:
            particles_cloud.point_data[scalars_name] = scalars[iter]

        if rigid_positions_stack is None:
            pl.write_frame()
            continue

        rigid_positions = rigid_positions_stack[iter]

        rigid_cloud.points = points_to_3D(rigid_positions, dim)

        pl.write_frame()

    pl.close()


def save_vtk(
    positions_stack,
    scalar_stack =None,
    scalar_name = "Scalar",
    output_folder= "./output/"
):
    num_output,num_points,dim = positions_stack.shape
    for pi,pos_stack in enumerate(positions_stack):
        particles_cloud = pv.PolyData( points_to_3D(pos_stack,dim))
        
        if scalar_stack is not None:
            particles_cloud.point_data[scalar_name] = scalar_stack[pi]

        particles_cloud.save(f"{output_folder}/particle_positions_{pi:04}.vtk")
        