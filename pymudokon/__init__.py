# """Pymudokon MPM library.

# Built with JAX.
# """

from .forces.dirichletbox import DirichletBox
from .forces.forces import Forces
from .forces.gravity import Gravity
from .forces.rigidparticles import RigidParticles

from .materials.modifiedcamclay import ModifiedCamClay
from .materials.mu_i_rheology import MuI
from .materials.druckerprager import DruckerPrager


from .materials.linearelastic import LinearIsotropicElastic
from .materials.newtonfluid import NewtonFluid
from .nodes.nodes import Nodes
from .particles.particles import Particles
from .shapefunctions.cubic import CubicShapeFunction
from .shapefunctions.linear import LinearShapeFunction
from .solvers.solver import run_solver
from .solvers.usl import USL
from .utils.domain import discretize
from .utils.io_plot import plot_simple
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

# # To be added soon
# # from .solvers.usl_apic import USL_APIC


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


from .utils.stl_helpers import sample_points_on_surface, get_stl_bounds, sample_points_in_volume
