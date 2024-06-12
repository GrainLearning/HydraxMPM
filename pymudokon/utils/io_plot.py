



import pyvista as pv
from typing import Dict, Any

import numpy as np


def points_to_3D(
        points,
        dim=2
    ):
    if dim == 3:
        return points
    return np.pad(points, [(0, 0), (0, 1)], mode="constant")


def plot_simple_3D(
        point_data_dict: Dict[str, np.ndarray],
        origin: np.ndarray,
        end: np.ndarray,
        output_file: str="output.gif",
        plot_params: Dict[str, Any] = None,
        camera_params: Dict[str, Any] = None
    ):

    num_iterations,num_points, dim = point_data_dict["points"].shape
    
    if "points" not in point_data_dict:
        raise ValueError("The point_data_dict must contain a key 'points' with the positions of the points")

    cloud = pv.PolyData(
        points_to_3D(point_data_dict["points"][0],dim)
        )

    for key, value in point_data_dict.items():
        if key == "points":
            continue

        cloud.point_data[key] = value[0]

    if dim == 2:
        origin = np.pad(origin, [(0, 1)], mode="constant")
        end = np.pad(end, [(0, 1)], mode="constant")

    box = pv.Box(bounds=np.array(list(zip(origin, end))).flatten())

    pl = pv.Plotter()

    if plot_params is None:
        plot_params = {}

    if "style" not in plot_params:
        plot_params["style"] = "points"
    
    if "cmap" not in plot_params:
        plot_params["cmap"] = "inferno"
    
    if "show_edges" not in plot_params:
        plot_params["show_edges"] = True

    if "clim" not in plot_params:
        plot_params["clim"] = [-0.1, 0.1]


    pl.add_mesh(
        cloud,
        **plot_params
    )

    pl.add_mesh(
        box,
        color="white",
        style="wireframe"
    )

    if camera_params is None:
        camera_params = {}

    if "camera_position" not in camera_params:
        if dim == 2:
            camera_params["camera_position"] = "xy"
        else:
            camera_params["camera_position"] = "xz"
    
    if "azimuth" not in camera_params:
        if dim == 2:
            camera_params["azimuth"] = 0
        else:
            camera_params["azimuth"] = 45
    
    if "elevation" not in camera_params:
        if dim == 2:
            camera_params["elevation"] = 0
        else:
            camera_params["elevation"] = 30
    
    if "zoom" not in camera_params:
        camera_params["zoom"] = 0.9
        
    

    pl.camera_position = camera_params["camera_position"]
    pl.camera.azimuth = camera_params["azimuth"]
    pl.camera.elevation = camera_params["elevation"]
    pl.camera.zoom(camera_params["zoom"])



    pl.open_gif(output_file)

    for iter in range(num_iterations):

        cloud.points = points_to_3D(point_data_dict["points"][iter],dim)
        for key, value in point_data_dict.items():
            if key == "points":
                continue
            cloud.point_data[key] = value[iter]

        pl.write_frame()

    pl.close()
