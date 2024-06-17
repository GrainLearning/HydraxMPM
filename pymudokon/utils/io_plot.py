import pyvista as pv
from typing import Dict, Any, List

import numpy as np


def points_to_3D(points, dim=2):
    if dim == 3:
        return points
    return np.pad(points, [(0, 0), (0, 1)], mode="constant")


def add_cloud_mesh(
    positions: np.ndarray, scalars: np.ndarray, scalar_name: str, params: Dict[str, Any], plotter: pv.Plotter
):
    num_points, dim = positions[0].shape

    cloud = pv.PolyData(points_to_3D(positions[0], dim))

    if scalars is not None:
        cloud.point_data[scalar_name] = scalars[0]

    if params is None:
        params = {}

    params.setdefault("style", "points")
    params.setdefault("cmap", "inferno")
    params.setdefault("show_edges", True)
    params.setdefault("clim", [-0.1, 0.1])
    params.setdefault("scalar_bar_args", dict(position_x=0.2, position_y=0.01))
    params.setdefault("scalars", scalar_name)

    plotter.add_mesh(cloud, **params)
    return plotter, cloud


def update_cloud_mesh(iter: int, positions: np.ndarray, scalars: np.ndarray, scalar_name: str, cloud: pv.PolyData):
    num_points, dim = positions[iter].shape
    cloud.points = points_to_3D(positions[iter], dim)
    if scalars is not None:
        cloud.point_data[scalar_name] = scalars[iter]
    return cloud


def add_box_mesh(origin: np.ndarray, end: np.ndarray, params: Dict[str, Any], plotter: pv.Plotter):
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
    particles_points: List[np.ndarray],
    particles_scalars: List[np.ndarray] = None,
    particles_scalar_name: str = None,
    rigid_points: List[np.ndarray] = None,
    box_plot_params: Dict[str, Any] = None,
    particles_plot_params: Dict[str, Any] = None,
    rigid_plot_params: Dict[str, Any] = None,
    theme: str = "dark",
    output_file: str = "output.gif",
):
    pv.set_plot_theme(theme)

    pl = pv.Plotter()
    pl, box = add_box_mesh(origin, end, box_plot_params, pl)

    pl, particles_cloud = add_cloud_mesh(
        particles_points, particles_scalars, particles_scalar_name, particles_plot_params, pl
    )

    if rigid_points is not None:
        pl, rigid_cloud = add_cloud_mesh(rigid_points, None, None, rigid_plot_params, pl)

    pl = set_camera(particles_plot_params, particles_points[0].shape[1], pl)

    pl.open_gif(output_file)

    for iter in range(len(particles_points)):
        particles_cloud = update_cloud_mesh(
            iter, particles_points, particles_scalars[iter], particles_scalar_name, particles_cloud
        )

        if rigid_points is not None:
            rigid_cloud = update_cloud_mesh(iter, rigid_points, None, None, rigid_cloud)
        pl.write_frame()

    pl.close()
