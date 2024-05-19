"""Pymudokon MPM library.

Built with JAX.
"""

# base classes
from .core.base import Base
from .core.interactions import Interactions
from .core.nodes import Nodes
from .core.particles import Particles
from .forces.forces import Forces
from .forces.dirichletbox import DirichletBox
from .materials.linearelastic import LinearIsotropicElastic
from .materials.material import Material
from .shapefunctions.cubic import CubicShapeFunction
from .shapefunctions.linear import LinearShapeFunction
from .shapefunctions.shapefunction import ShapeFunction
from .solvers.solver import Solver
from .solvers.usl import USL

from .utils.plot import create_plotter, update_plotter

__all__ = [
    "Interactions",
    "Nodes",
    "Particles",
    "LinearIsotropicElastic",
    "CubicShapeFunction",
    "LinearShapeFunction",
    "USL",
    "Forces",
    "DirichletBox",
    "Solver",
    "Material",
    "Base",
    "ShapeFunction",
    "create_plotter",
    "update_plotter"
]
