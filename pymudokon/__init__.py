"""Pymudokon MPM library.

Built with JAX.
"""

# core
from .core.base import Base
from .core.interactions import Interactions
from .core.nodes import Nodes
from .core.particles import Particles
# forces
from .forces.forces import Forces
from .forces.dirichletbox import DirichletBox
from .forces.gravity import Gravity
# materials
from .materials.material import Material
from .materials.newtonfluid import NewtonFluid
from .materials.linearelastic import LinearIsotropicElastic
from .materials.modifiedcamclay import ModifiedCamClay
# shapefunctions
from .shapefunctions.shapefunction import ShapeFunction
from .shapefunctions.cubic import CubicShapeFunction
from .shapefunctions.linear import LinearShapeFunction
# solvers
from .solvers.solver import Solver
from .solvers.usl import USL
# utils
from .utils.plot import (
    plotter_add_nodes,
    plotter_add_particles,
    plotter_create,
    plotter_update_particles,
    plotter_update_nodes,
    update_plotter_animation,
)
# single element benchmarks
from .ip_benchmarks.simpleshear import simple_shear

from .ip_benchmarks.plotting import plot_tau_gamma,plot_q_p

__all__ = [
    "Interactions",
    "Nodes",
    "Particles",
    "LinearIsotropicElastic",
    "NewtonFluid",
    "ModifiedCamClay",
    "CubicShapeFunction",
    "LinearShapeFunction",
    "USL",
    "Forces",
    "DirichletBox",
    "Gravity",
    "Solver",
    "Material",
    "Base",
    "ShapeFunction",
    "plotter_create",
    "plotter_add_particles",
    "plotter_add_nodes",
    "plotter_update_particles",
    "plotter_update_nodes",
    "update_plotter_animation",
    "simple_shear",
    "plot_tau_gamma",
    "plot_q_p"
]
