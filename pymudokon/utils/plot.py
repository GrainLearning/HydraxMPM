import pyvista as pv
import dataclasses
from typing import List, Dict
import dataclasses
from pymudokon import Solver
from jax import numpy as jnp
import pyvista as pv

def create_plotter(usl:Solver,pl=None):
    """
    Temporary function to create plotter
    """
    points = usl.particles.positions
    points_3d = jnp.pad(points, [(0, 0), (0, 1)], mode="constant").__array__()
    
    velocities = usl.particles.velocities
    mean_velocities = velocities.mean(axis=1)
    
 
    # Create a PolyData object
    cloud = pv.PolyData(points_3d)
    cloud.point_data["mean_velocities"] = mean_velocities
    
    
    pl.add_mesh(
        cloud,
        scalars="mean_velocities",
        style="points", 
        emissive=False,
        show_edges=True,
        render_points_as_spheres=True,
        cmap='inferno',
        point_size=20.0)
    
    pl.camera_position = 'xy'
    
    pl.open_gif("out.gif")

    return pl, cloud

def update_plotter(pl, cloud, usl:Solver):
    """
    Temporary function to update plotter
    """
    points = usl.particles.positions
    velocities = usl.particles.velocities
    points_3d = jnp.pad(points, [(0, 0), (0, 1)], mode="constant").__array__()
    
    cloud["mean_velocities"] = velocities.mean(axis=1)
    cloud.points = points_3d
    pl.render()

    pl.write_frame()
    return pl,cloud
