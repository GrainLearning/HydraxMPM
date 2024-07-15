import vtk
import pyvista as pv
import numpy as np


def sample_points_on_surface(mesh_path, distance=0.001, plot=False):
    reader = pv.get_reader(mesh_path)

    mesh = reader.read()

    sampler = vtk.vtkPolyDataPointSampler()

    sampler.SetInputData(mesh)

    sampler.SetDistance(distance)

    sampler.SetPointGenerationModeToRandom()

    sampler.Update()

    point_cloud = pv.wrap(sampler.GetOutput())

    if plot:
        pl = pv.Plotter()

        pl.add_mesh(point_cloud, color="tan", show_edges=True)

        pl.add_mesh(mesh, color="red", show_edges=True)

        pl.show()

    return np.array(point_cloud.points)


def get_stl_bounds(mesh_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()

    bounds = reader.GetOutput().GetBounds()

    return np.array(bounds).reshape(3, 2).T


def sample_points_in_volume(mesh_path, num_points=1000):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(mesh_path)
    reader.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.FlipNormalsOn()
    normals.Update()

    geometry = normals.GetOutput()

    np.random.seed(4355412)

    bounds = geometry.GetBounds()

    print(f"Bounds uniform random grid: {bounds[0]}, {bounds[1]} {bounds[2]}, {bounds[3]} {bounds[4]}, {bounds[5]}")

    # Generate random points within the bounding box of the polydata
    points = vtk.vtkPoints()
    pointsPolyData = vtk.vtkPolyData()
    pointsPolyData.SetPoints(points)

    points.SetNumberOfPoints(num_points)
    for i in range(num_points):
        point = [
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[2], bounds[3]),
            np.random.uniform(bounds[4], bounds[5]),
        ]
        points.SetPoint(i, point[0], point[1], point[2])

    extract = vtk.vtkExtractEnclosedPoints()
    extract.SetSurfaceData(geometry)
    extract.SetInputData(pointsPolyData)
    extract.SetTolerance(0.001)
    extract.CheckSurfaceOn()
    extract.Update()

    positions_cpu = []
    for id in range(num_points):
        point = extract.GetOutput().GetPoint(id)
        positions_cpu.append([point[0], point[1], point[2]])

    return positions_cpu
