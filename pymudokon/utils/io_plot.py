



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


def add_point_mesh(
    positions: np.ndarray,
    scalars: np.ndarray,
    scalar_name: str,
    params: Dict[str, Any],
    plotter: pv.Plotter
    ):
    num_points, dim = positions.shape

    cloud = pv.PolyData(points_to_3D(positions,dim))

    cloud.point_data[scalar_name] = scalars

    plotter.add_mesh(
        cloud,
        **params
    )
    return plotter, cloud


def plot_simple(
        particles_positions: np.ndarray,
        particles_scalars: np.ndarray,
        particles_scalar_name: str,
        theme: str = "dark",
    ):
    pv.set_plot_theme(theme)
    pl = pv.Plotter()

    pl, cloud = add_point_mesh(
        particles_positions,
        particles_scalars,
        particles_scalar_name,
        dict(style="points", cmap="inferno", show_edges=True, clim=[-0.1, 0.1], scalar_bar_args=dict(position_x=0.2, position_y=0.01)),
    )
# def update_point_mesh(
#         positions: np.ndarray,
#         scalars: np.ndarray,
#         scalar_name: str,
#         plotter: pv.Plotter
# )


# def plot_simple_3D(
    #     point_data_dict: Dict[str, np.ndarray],
    #     origin: np.ndarray,
    #     end: np.ndarray,
    #     output_file: str="output.gif",
    #     plot_params: Dict[str, Any] = None,
    #     camera_params: Dict[str, Any] = None,
    #     theme="dark",
    #     points_data_rigid_dict: Dict[str, np.ndarray] = None,
    #     plot_params_rigid: Dict[str, Any] = None
    # ):
    # 
    # num_iterations,num_points, dim = point_data_dict["points"].shape
    
    # if "points" not in point_data_dict:
    #     raise ValueError("The point_data_dict must contain a key 'points' with the positions of the points")



    # for key, value in point_data_dict.items():
    #     if key == "points":
    #         continue

    #     cloud.point_data[key] = value[0]

    # if dim == 2:
    #     origin = np.pad(origin, [(0, 1)], mode="constant")
    #     end = np.pad(end, [(0, 1)], mode="constant")

    # box = pv.Box(bounds=np.array(list(zip(origin, end))).flatten())

    # pl = pv.Plotter()
    
    # if plot_params is None:
    #     plot_params = {}

    # if "style" not in plot_params:
    #     plot_params["style"] = "points"
    
    # if "cmap" not in plot_params:
    #     plot_params["cmap"] = "inferno"
    
    # if "show_edges" not in plot_params:
    #     plot_params["show_edges"] = True

    # if "clim" not in plot_params:
    #     plot_params["clim"] = [-0.1, 0.1]

    # if "scalar_bar_args" not in plot_params:
    #     plot_params["scalar_bar_args"] = dict( position_x=0.2, position_y=0.01)
    # pl.add_mesh(
    #     cloud,
    #     **plot_params
    # )

    # pl.add_mesh(
    #     box,
    #     color="white",
    #     style="wireframe"
    # )

    # if points_data_rigid_dict is not None:
    #     cloud_rigid = pv.PolyData(
    #         points_to_3D(points_data_rigid_dict["points"][0],dim)
    #     )
    #     for key, value in points_data_rigid_dict.items():
    #         if key == "points":
    #             continue

    #     cloud_rigid.point_data[key] = value[0]

    #     if plot_params_rigid is None:
    #         plot_params_rigid = {}
    #     if "style" not in plot_params_rigid:
    #         plot_params_rigid["style"] = "points"
            
    #     pl.add_mesh(
    #         cloud_rigid,
    #         **plot_params_rigid
    #     )
    

    # if camera_params is None:
    #     camera_params = {}

    # if "camera_position" not in camera_params:
    #     if dim == 2:
    #         camera_params["camera_position"] = "xy"
    #     else:
    #         camera_params["camera_position"] = "xz"
    
    # if "azimuth" not in camera_params:
    #     if dim == 2:
    #         camera_params["azimuth"] = 0
    #     else:
    #         camera_params["azimuth"] = 45
    
    # if "elevation" not in camera_params:
    #     if dim == 2:
    #         camera_params["elevation"] = 0
    #     else:
    #         camera_params["elevation"] = 30
    
    # if "zoom" not in camera_params:
    #     if dim == 2:
    #         camera_params["zoom"] = 1.1
    #     else:
    #         camera_params["zoom"] = 0.9
        
    

    # pl.camera_position = camera_params["camera_position"]
    # pl.camera.azimuth = camera_params["azimuth"]
    # pl.camera.elevation = camera_params["elevation"]
    # pl.camera.zoom(camera_params["zoom"])



    # pl.open_gif(output_file)

    # for iter in range(num_iterations):

    #     cloud.points = points_to_3D(point_data_dict["points"][iter],dim)


    #     if "scalars" in plot_params:
    #         cloud.point_data[plot_params["scalars"]] = point_data_dict[plot_params["scalars"]][iter]

    #     pl.write_frame()

    # pl.close()
