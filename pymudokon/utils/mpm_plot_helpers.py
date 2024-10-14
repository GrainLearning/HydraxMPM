import dataclasses
from typing import Dict, List, Tuple

import numpy as np
import pyvista as pv


def points_to_3D(points, dim=2):
    """Convert 2D points to 3D points."""
    if dim == 3:
        return np.array(points)
    return np.pad(points, [(0, 0), (0, 1)], mode="constant")

@dataclasses.dataclass
class PvPointHelper:
    timeseries: pv.MultiBlock = None
    timeseries_options: Dict = None
    bbox: pv.Box =None
    bbox_options: Dict = None
    subplot: Tuple = None
    
    @classmethod
    def create(cls,
               position_stack,
               origin,
               end,
               subplot = None,
               scalar_stack=None,
               scalar_name = "scalar",
               store_bbox=True,
               bbox_options = None,
               timeseries_options = None,
               camera_options = None
               ):
        
        if bbox_options is None:
            bbox_options = {}
            
        bbox_options.setdefault("color", "black")
        bbox_options.setdefault("style", "wireframe")
        
        if timeseries_options is None:
            timeseries_options = {}
            

        if scalar_stack is not None:
            timeseries_options.setdefault("scalars",scalar_name)
        
        num_frames, num_points, dim = position_stack.shape
        if dim == 3:
            position_stack = np.array(position_stack)
        elif dim == 2:
            position_stack = np.pad(position_stack, [(0,0),(0, 0), (0, 1)], mode="constant")
            origin = np.pad(origin,(0,1))
            end = np.pad(end,(0,1))
        
        

        
        timeseries = pv.MultiBlock()
        for ti, positions in enumerate(position_stack):
            polydata =  pv.PolyData(positions)
            
            if scalar_stack is not None:
                polydata.point_data[scalar_name] = scalar_stack[ti]
                
            timeseries.append(polydata)
        
        if store_bbox:
            bbox = pv.Box(bounds=np.array(list(zip(origin, end))).flatten())
        else:
            bbox = None
            
        if subplot is None:
            subplot = (0,0)
            
        return cls(
            timeseries =timeseries,
            timeseries_options=timeseries_options,
            bbox = bbox,
            subplot = subplot,
            bbox_options = bbox_options
        )
    
def make_pvplots(
    mpmplot_list: List[PvPointHelper]= None,
    file: str = "output.gif",
    plotter_options: Dict = None,
    plotter:pv.Plotter=None,
    camera_options = None,
    dim= 3
):

    if plotter_options is None:
        plotter_options = {}
        
    plotter_options.setdefault("shape", (1,1))

    if plotter is None:
        plotter = pv.Plotter(notebook=False, off_screen=True,**plotter_options)
    
    if camera_options is None:
        camera_options = {}
        if dim == 2:
            camera_options.setdefault("camera_position", "xy")
            # camera_options.setdefault("azimuth", 0)
            # camera_options.setdefault("elevation", 0)
            # camera_options.setdefault("zoom", 0.98)
        else:
            camera_options.setdefault("camera_position", "xz")
            camera_options.setdefault("azimuth", 45)
            camera_options.setdefault("elevation", 30)
            camera_options.setdefault("zoom", 0.9)
    
    plotter.link_views()
    



                        
    plotter.open_gif(file)
    num_frames = len(mpmplot_list[0].timeseries)
    

    plotter.clear()
    for frame in range(num_frames):

        for si in range(plotter_options["shape"][0]):
            for sj in range(plotter_options["shape"][1]):
                plotter.subplot(si,sj)
            
                for mpmplot in mpmplot_list:
                    if mpmplot.subplot != (si,sj):
                        continue
                    

                    
                    plotter.add_mesh(
                        mpmplot.timeseries[frame],
                        **mpmplot.timeseries_options
                    )
                    
                    plotter.add_mesh(
                        mpmplot.bbox,
                        **mpmplot.bbox_options
                    )
                    # plotter.set_focus(mpmplot.timeseries[frame].points[1])
                    if frame==0:
                        plotter.camera.tight(padding=0.10,adjust_render_window=True)
                    # for key, value in camera_options.items():
                        # setattr(plotter.camera, key, value)
                    # plotter.camera.azimuth = camera_options["azimuth"]
                    # plotter.camera.elevation = camera_options["elevation"]
                    # plotter.camera.zoom(camera_options["zoom"])
        
        plotter.write_frame()
        plotter.clear()
                
    plotter.close()
    
    

        
    # for unique_subplot in list(set(subplots)):
    #     for  si,subplot in enumerate(subplots):
    #         if unique_subplot == subplot:
                
    #             for timeseries 
    #             plotter.add_mesh(mpmplot_list[si].timeseries, **mpmplot_list[si].timeseries.timeseries_options)
                
            

#     # initialize
#     for si, mpmplot in enumerate(mpmplot_list):
        
#         index_row, index_col = subplots[si]
        
#         plotter.subplot(index_row,index_col)
        
#         cloud = pv.PolyData(mpmplot.position_stack[0])
        
#         if mpmplot.scalar_name is not None:
#             cloud.point_data[mpmplot.scalar_name] = mpmplot.scalar_stack[0]
        


#         plotter.add_mesh(box, **box_options)


            
#     # Set Camera


#     for si, mpmplot in enumerate(mpmplot_list):
        
#         index_row, index_col = subplots[si]
#         plotter.subplot(index_row,index_col)
#         plotter.camera_position = params["camera_position"]
# #     plotter.camera.azimuth = params["azimuth"]
# #     plotter.camera.elevation = params["elevation"]
# #     plotter.camera.zoom(params["zoom"])



#     return plotter
# # def add_box_mesh(
# #     origin: np.ndarray, end: np.ndarray, params: Dict[str, Any], plotter: pv.Plotter
# # ):
# #     """Add box mesh to plotter."""


# #     b






    

#     create

# def add_cloud_mesh(
#     positions: np.ndarray,
#     scalars: np.ndarray,
#     scalar_name: str,
#     params: Dict[str, Any],
#     plotter: pv.Plotter,
# ):
#     """Add point cloud to plotter."""
#     num_points, dim = positions[0].shape

#     cloud = pv.PolyData(points_to_3D(positions[0], dim))

#     if scalars is not None:
#         cloud.point_data[scalar_name] = scalars[0]

#     if params is None:
#         params = {}

#     params.setdefault("show_edges", True)
#     params.setdefault("style", "points")
#     # params.setdefault("cmap", "viridis")
#     # params.setdefault("clim", [-0.1, 0.1])
#     # params.setdefault("scalar_bar_args", dict(position_x=0.2, position_y=0.01))
#     # params.setdefault("scalars", scalar_name)

#     plotter.add_mesh(cloud, **params)
#     return plotter, cloud



# def set_camera(params: Dict[str, Any], dim: int, plotter: pv.Plotter):
#     """Set camera position and zoom for 2D or 3D plots."""



# def plot_simple(
#     origin: np.ndarray,
#     end: np.ndarray,
#     positions_stack: List[np.ndarray],
#     scalars: List[np.ndarray] = None,
#     scalars_name: str = None,
#     rigid_positions_stack: List[np.ndarray] = None,
#     box_plot_params: Dict[str, Any] = None,
#     particles_plot_params: Dict[str, Any] = None,
#     rigid_plot_params: Dict[str, Any] = None,
#     theme: str = "default",
#     output_file: str = "output.gif",
#     highlight_indices= None
# ):
#     """Pyvista wrapper to plot MPM simulation.

#     Args:
#         origin (np.ndarray): Domain origin
#         end (np.ndarray): Domain end
#         positions_stack (List[np.ndarray]): List of particle positions
#         scalars (List[np.ndarray], optional): List of scalars. Defaults to None.
#         scalars_name (str, optional): Name of scalar. Defaults to None.
#         rigid_positions_stack (List[np.ndarray], optional):
#             List of rigid particle positions. Defaults to None.
#         box_plot_params (Dict[str, Any], optional):
#             Plot params of domain boundary. Defaults to None.
#         particles_plot_params (Dict[str, Any], optional):
#             Plot params of particles. Defaults to None.
#         rigid_plot_params (Dict[str, Any], optional):
#             Plot params of rigid particles. Defaults to None.
#         theme (str, optional): Pyvista theme to plot. Defaults to "default".
#         output_file (str, optional): Output file name. Defaults to "output.gif".
#     """
#     pv.set_plot_theme(theme)

#     pl = pv.Plotter(notebook=False, off_screen=True)
#     pl, box = add_box_mesh(origin, end, box_plot_params, pl)

#     pl, particles_cloud = add_cloud_mesh(
#         positions_stack, scalars, scalars_name, particles_plot_params, pl
#     )
    
#     if highlight_indices is not None:    
#         pl, highlight_cloud = add_cloud_mesh(
#             positions_stack.at[:,highlight_indices].get(), None, None, 
#             {
#             #  "zorder":10,
#             # "cmap":"tab20",
#             # "render_points_as_spheres":True,
#              "point_size":10
#             },
#             pl
#         )
        
#         cmap = cm.get_cmap("tab20")  # Choose a colormap with enough colors
#         colors = cmap(np.arange(highlight_cloud.n_points))
#         highlight_cloud.point_data["colors"] = colors
#         highlight_cloud.set_active_scalars("colors")
    

    
#     if rigid_positions_stack is not None:
#         pl, rigid_cloud = add_cloud_mesh(
#             rigid_positions_stack, None, None, rigid_plot_params, pl
#         )

#     pl = set_camera(particles_plot_params, positions_stack[0].shape[1], pl)

#     pl.open_gif(output_file)


#     for iter in range(len(positions_stack)):
#         positions = positions_stack[iter]

#         num_points, dim = positions.shape

#         particles_cloud.points = points_to_3D(positions, dim)

#         if scalars is not None:
#             particles_cloud.point_data[scalars_name] = scalars[iter]

#         if highlight_indices is not None:
#             highlight_cloud.points = points_to_3D(positions.at[highlight_indices].get(), dim)


#         if rigid_positions_stack is None:
#             pl.write_frame()
#             continue

#         rigid_positions = rigid_positions_stack[iter]

#         rigid_cloud.points = points_to_3D(rigid_positions, dim)

#         pl.write_frame()

#     pl.close()


# def save_vtk(
#     positions_stack,
#     scalar_stack =None,
#     scalar_name = "Scalar",
#     output_folder= "./output/"
# ):
#     num_output,num_points,dim = positions_stack.shape
#     for pi,pos_stack in enumerate(positions_stack):
#         particles_cloud = pv.PolyData( points_to_3D(pos_stack,dim))
        
#         if scalar_stack is not None:
#             particles_cloud.point_data[scalar_name] = scalar_stack[pi]

#         particles_cloud.save(f"{output_folder}/particle_positions_{pi:04}.vtk")
        