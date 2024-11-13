import numpy as np
import vtk


def sample_points_on_surface(mesh_path, distance=0.001, plot=False):
    import pyvista as pv

    reader = pv.get_reader(mesh_path)

    mesh = reader.read()

    sampler = vtk.vtkPolyDataPointSampler()

    sampler.SetInputData(mesh)

    sampler.SetDistance(distance)

    sampler.SetPointGenerationModeToRandom()

    sampler.Update()

    point_cloud = pv.wrap(sampler.GetOutput())

    return np.array(point_cloud.points)


def get_stl_bounds(mesh_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()

    bounds = reader.GetOutput().GetBounds()

    return np.array(bounds).reshape(3, 2).T


def sample_points_in_volume(
    mesh_path: str,
    num_points: int = 1000,
    points: np.array = None,
    return_surface: bool = False,
):
    import pyvista as pv

    surface = pv.read(mesh_path)
    if points is None:
        origin, end = get_stl_bounds(mesh_path)
        x = np.random.uniform(origin[0], end[0], num_points)
        y = np.random.uniform(origin[1], end[1], num_points)
        z = np.random.uniform(origin[2], end[2], num_points)
        points = np.vstack((x, y, z)).T
    else:
        points = np.array(points)

    # Create a PolyData object for the points
    point_cloud = pv.PolyData(points)

    # Use select_enclosed_points to filter points inside the surface
    enclosed_points = point_cloud.select_enclosed_points(
        surface, tolerance=0.00001, inside_out=False, check_surface=True
    )

    # Extract the points that are inside the surface
    inside_points = enclosed_points.points[
        enclosed_points.point_data["SelectedPoints"] == 1
    ]

    if return_surface:
        return inside_points, surface
    return inside_points
