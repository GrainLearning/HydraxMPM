"""Pymudokon MPM library.

Built with JAX.
"""

# core
from .core.nodes import Nodes

from .core.particles import Particles

from .forces.dirichletbox import DirichletBox
from .forces.forces import Forces

from .forces.gravity import Gravity
from .forces.nodewall import NodeWall

from .forces.rigidparticles import RigidParticles

# These will soon be added back
# from .ip_benchmarks.simpleshear import simple_shear
# from .solvers.usl_apic import USL_APIC
# from .ip_benchmarks.triaxial import triaxial_compression
# from .materials.modifiedcamclay import ModifiedCamClay


from .materials.linearelastic import LinearIsotropicElastic
from .materials.material import Material


from .materials.newtonfluid import NewtonFluid
from .shapefunctions.cubic import CubicShapeFunction
from .shapefunctions.linear import LinearShapeFunction
from .shapefunctions.shapefunction import ShapeFunction
from .solvers.solver import Solver
from .solvers.usl import USL

from .utils.domain import discretize
from .utils.io_plot import plot_simple, points_to_3D
from .utils.math_helpers import (
    get_dev_strain,
    get_dev_stress,
    get_gamma,
    get_KE,
    get_pressure,
    get_q_vm,
    get_tau,
    get_volumetric_strain,
)

__all__ = [
    "ShapeFunction",
    "LinearShapeFunction",
    "Nodes",
    "Particles",
    "LinearIsotropicElastic",
    "NewtonFluid",
    "CubicShapeFunction",
    "USL",
    "Forces",
    "DirichletBox",
    "Gravity",
    "Solver",
    "Material",
    "simple_shear",
    "triaxial_compression",
    "discretize",
    "get_dev_stress",
    "get_dev_strain",
    "get_gamma",
    "get_q_vm",
    "get_tau",
    "get_volumetric_strain",
    "get_KE",
    "get_pressure",
    "plot_simple",
    "points_to_3D",
    "NodeWall",
    "RigidParticles",
    # "USL_APIC",
    # "ModifiedCamClay",
]
