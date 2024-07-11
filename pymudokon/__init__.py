"""Pymudokon MPM library.

Built with JAX.
"""

# To be added soon
# from .solvers.usl_apic import USL_APIC
# from .materials.modifiedcamclay import ModifiedCamClay

# core
from .core.nodes import Nodes
from .core.particles import Particles
from .forces.dirichletbox import DirichletBox
from .forces.forces import Forces
from .forces.gravity import Gravity
from .forces.nodewall import NodeWall
from .forces.rigidparticles import RigidParticles
from .materials.druckerprager import DruckerPrager
from .materials.linearelastic import LinearIsotropicElastic
from .materials.material import Material
from .materials.mu_i_rheology import MuI
from .materials.newtonfluid import NewtonFluid
from .materials_analysis.deformation_wrappers import (
    simple_shear_wrapper,
    triaxial_compression_wrapper,
    isotropic_compression_wrapper,
)
from .materials_analysis.mix_control import mix_control
from .materials_analysis.plot import (
    plot_p_dot_gamma,
    plot_p_gamma,
    plot_q_dot_gamma,
    plot_q_gamma,
    plot_q_p,
    plot_strain_grid,
    plot_stress_grid,
    plot_suite,
)
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
