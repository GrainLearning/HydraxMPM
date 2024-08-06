"""Work in progress example of a flodex simulation"""

import pyvista as pv

import vtk

import numpy as np

import os

import pymudokon as pm

project_path = os.path.dirname(os.path.realpath(__file__)) + "/"

geometry_file_path = "flodex.stl"

domain_file = "domain.stl"

material_file = "material.stl"

rigid_particles = pm.sample_points_on_surface(geometry_file_path, distance=0.001)

origin, end = pm.get_stl_bounds(domain_file)


points = pm.sample_points_in_volume(material_file, num_points=1000)
