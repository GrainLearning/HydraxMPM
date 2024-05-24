"""Pymudokon MPM library.

Built with JAX.
"""

# core
from .core.base import Base
from .core.interactions import Interactions
from .core.nodes import Nodes
from .core.particles import Particles
from .forces.dirichletbox import DirichletBox
from .forces.forces import Forces
from .forces.gravity import Gravity
from .ip_benchmarks.plotting import plot_q_p, plot_tau_gamma
from .ip_benchmarks.simpleshear import simple_shear
from .materials.linearelastic import LinearIsotropicElastic
from .materials.material import Material
from .materials.modifiedcamclay import ModifiedCamClay
from .materials.newtonfluid import NewtonFluid
from .shapefunctions.cubic import CubicShapeFunction
from .shapefunctions.linear import LinearShapeFunction
from .solvers.solver import Solver
from .solvers.usl import USL
from .utils.domain import discretize

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
    "simple_shear",
    "plot_tau_gamma",
    "plot_q_p",
    "discretize",
]
