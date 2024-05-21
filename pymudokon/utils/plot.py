from typing import Tuple

import numpy as np
import pyvista as pv
from jax import numpy as jnp

from pymudokon import Nodes, Particles


def plotter_create(filepath="out.gif", shape: Tuple = (1, 1), bounds=None) -> pv.Plotter:
    """Create a plotter for 2D simulation.

    Args:
        filepath (str, optional):
            Path of output. Defaults to "out.gif".
        shape (Tuple, optional):
            Shape of the plotter. Defaults to (1, 1).

    Returns:
        pv.Plotter: PyVista plotter object
    """
    pl = pv.Plotter(shape=shape)

    if bounds is not None:
        box = pv.Box(bounds=bounds)

        pl.add_mesh(
            box,
            show_edges=True,
            opacity=1.0,
            color="white",
        )

    pl.camera_position = "xy"
    pl.open_gif(filepath)
    return pl


def plotter_add_particles(pl: pv.Plotter, particles: Particles, clim=None,point_size=10.0) -> Tuple[pv.Plotter, pv.PolyData]:
    """Add particles to the plotter.

    Args:
        pl (pv.Plotter):
            PyVista plotter object.
        particles (Particles):
            MPM particles.

    Returns:
        Tuple[pv.Plotter, pv.PolyData]:
            PyVista plotter object and PolyData object.
    """
    points_3d = jnp.pad(particles.positions, [(0, 0), (0, 1)], mode="constant").__array__()
    mean_velocities = particles.velocities.mean(axis=1)

    # Create a PolyData object
    cloud = pv.PolyData(points_3d)
    cloud.point_data["mean_velocities"] = mean_velocities

    pl.add_mesh(
        cloud,
        scalars="mean_velocities",
        style="points",
        show_edges=True,
        render_points_as_spheres=True,
        cmap="inferno",
        point_size=point_size,
        clim=clim,
    )
    pl.camera_position = "xy"
    return pl, cloud


def plotter_add_nodes(pl: pv.Plotter, nodes: Nodes) -> Tuple[pv.Plotter, pv.PolyData]:
    """Add nodes to the plotter.

    Args:
        pl (pv.Plotter):
            PyVista plotter object.
        nodes (Nodes):
            MPM nodes.

    Returns:
        Tuple[pv.Plotter, pv.PolyData]:
            PyVista plotter object and PolyData object.
    """
    xgrid = np.arange(nodes.origin[0], nodes.end[1] + nodes.node_spacing, nodes.node_spacing)

    ygrid = np.arange(nodes.origin[1], nodes.end[1] + nodes.node_spacing, nodes.node_spacing)

    grid = pv.RectilinearGrid(xgrid, ygrid)
    moments_3d = jnp.pad(nodes.moments, [(0, 0), (0, 1)], mode="constant").__array__()
    grid.point_data["Moments"] = moments_3d
    arrows = grid.glyph(scale="Moments", orient="Moments", tolerance=0.05)

    pl.add_mesh(grid, color="white", show_edges=True, opacity=0.5)
    pl.add_mesh(arrows, color="black")
    pl.camera_position = "xy"
    return pl, grid


# # Create a PolyData object
# cloud = pv.PolyData(points_3d)

# pl.add_mesh(
#     cloud,
#     style="points",
#     show_edges=True,
#     render_points_as_spheres=True,
#     point_size=20.0,
# )
#
# return pl, cloud


def plotter_update_particles(cloud: pv.PolyData, particles: Particles) -> pv.PolyData:
    """Update particles in the plotter.

    Args:
        pl (pv.Plotter):
            PyVista plotter object.
        cloud (pv.PolyData):
            PyVista PolyData object.
        particles (Particles):
            MPM particles.

    Returns:
        pv.PolyData:
            PolyData object.
    """
    points = particles.positions
    points_3d = jnp.pad(points, [(0, 0), (0, 1)], mode="constant").__array__()
    cloud.points = points_3d

    velocities = particles.velocities
    cloud["mean_velocities"] = velocities.mean(axis=1)

    return cloud


def plotter_update_nodes(grid: pv.PolyData, nodes: Nodes) -> pv.PolyData:
    """Update nodes in the plotter.

    Args:
        pl (pv.Plotter):
            PyVista plotter object.
        grid (pv.PolyData):
            PyVista PolyData object.
        nodes (Nodes):
            MPM nodes.

    Returns:
        pv.PolyData:
            PolyData object.
    """
    moments = nodes.moments
    moments_3d = jnp.pad(moments, [(0, 0), (0, 1)], mode="constant").__array__()
    grid.point_data["Moments"] = moments_3d * 10

    return grid


def update_plotter_animation(pl: pv.Plotter) -> pv.Plotter:
    """Update animation frame of the plotter.

    Args:
        pl (pv.Plotter):
            PyVista plotter object.

    Returns:
        pv.Plotter:
            Updated PyVista plotter object.
    """
    pl.render()

    pl.write_frame()
    return pl
