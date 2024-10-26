"""Pymudokon MPM library.

Built with JAX.
"""

# Future

# from .materials.mu_i_softness import MuISoft
# from .materials.UH_model import UHModel

from .utils.jax_helpers import dump_restart_files, save_object, set_default_gpu


from .config.mpm_config import MPMConfig

from .partition.grid_stencil_map import GridStencilMap
from .solvers.usl import USL
from .particles.particles import Particles
from .nodes.nodes import Nodes
from .shapefunctions.linear import LinearShapeFunction

from .materials.linearelastic import LinearIsotropicElastic

from .utils.mpm_domain_helpers import discretize, fill_domain_with_particles

from .materials_analysis.plot import (
    add_plot,
    add_plot_3d,
    make_plots,
    make_plots_3d,
    PlotHelper,
)
from .solvers.run_solver import run_solver
# run_solver_io



# from .utils.jax_helpers import create_shapefunction
# from .utils.jax_helpers import create_grid_partition

# from .forces.dirichletbox import DirichletBox
# from .forces.forces import Forces
# from .forces.gravity import Gravity
# from .forces.nodelevelset import NodeLevelSet
# from .forces.nodewall import NodeWall
# from .forces.rigidparticles import RigidParticles
# from .materials.druckerprager import DruckerPrager
# from .materials.experimental.mcc_mrm import MCC_MRM
# from .materials.experimental.mcc_reg import ModifiedCamClayReg
# from .materials.experimental.mrm_steadystate import MRMSteady
# from .materials.experimental.uh_model import UHModel
# from .materials.experimental.mcc_mrm_reg import MCC_MRM_Reg

# from .materials.material import Material
# from .materials.modifiedcamclay import ModifiedCamClay
# from .materials.mu_i_rheology_incompressible import MuI_incompressible
# from .materials.newtonfluid import NewtonFluid
# from .materials_analysis.mix_control import mix_control
# from .materials_analysis.mp_benchmarks import MPBenchmark

# from .materials_analysis.plot_sets import plot_set1, plot_set2, plot_set3
# from .nodes.nodes import Nodes

# from .nodes.nodes_grid import NodesGrid




# from .shapefunctions.cubic import CubicShapeFunction
# from .shapefunctions.cubic_old import CubicShapeFunction2
# # from .shapefunctions.linear import LinearShapeFunction
# from .shapefunctions.linear_grid import LinearShapeFunction
# from .shapefunctions.shapefunctions import ShapeFunction

# from .solvers.usl_grid import USLGrid
# from .partition.grid_partition import GridPartition
# from .solvers.usl_apic import USL_APIC
# from .solvers.usl_asflip import USL_ASFLIP

from .utils.math_helpers import (
#     e_to_phi,
#     e_to_phi_stack,
#     get_dev_strain,
#     get_dev_strain_stack,
#     get_dev_stress,
#     get_dev_stress_stack,
#     get_e_from_bulk_density,
#     get_hencky_strain_stack,
#     get_inertial_number,
#     get_inertial_number_stack,
#     get_J2,
#     get_J2_stack,
#     get_k0_stress,
#     get_KE,
    get_KE_stack,
#     get_phi_from_bulk_density,
#     get_phi_from_bulk_density_stack,
#     get_phi_from_L,
#     get_plastic_strain,
#     get_plastic_strain_stack,
#     get_pressure,
#     get_pressure_stack,
#     get_q_vm,
#     get_q_vm_stack,
#     get_scalar_shear_strain,
#     get_scalar_shear_strain_stack,
#     get_scalar_shear_stress,
#     get_scalar_shear_stress_stack,
#     get_skew_tensor,
#     get_skew_tensor_stack,
#     get_small_strain,
#     get_small_strain_stack,
#     get_strain_rate_from_L,
#     get_strain_rate_from_L_stack,
#     get_sym_tensor,
#     get_sym_tensor_stack,
#     get_volumetric_strain,
#     get_volumetric_strain_stack,
#     phi_to_e,
#     phi_to_e_stack,
)

from .utils.mpm_plot_helpers import make_pvplots, points_to_3D, PvPointHelper
# from .utils.mpm_postprocessing_helpers import (
#     post_processes_grid_gradient_stack,
#     post_processes_stress_stack,
# )
# from .utils.stl_helpers import (
#     get_stl_bounds,
#     sample_points_in_volume,
#     sample_points_on_surface,
# )


# __all__ = [
#     "Nodes",
#     "Particles",
#     "ShapeFunction",
#     "LinearShapeFunction",
#     "CubicShapeFunction",
#     "DirichletBox",
#     "RigidParticles",
#     "Forces",
#     "Gravity",
#     "NodeWall",
#     "Material",
#     "LinearIsotropicElastic",
#     "NewtonFluid",
#     "DruckerPrager",
#     "ModifiedCamClay",
#     "MCC_MRM",
#     "MuI",
#     "mix_control",
#     "MPBenchmark",
#     "USL",
#     "USL_APIC",
#     "run_solver",
#     "discretize",
#     "e_to_phi",
#     "e_to_phi_stack",
#     "get_dev_strain",
#     "get_dev_strain_stack",
#     "get_dev_stress",
#     "get_dev_stress_stack",
#     "get_e_from_bulk_density",
#     "get_inertial_number",
#     "get_inertial_number_stack",
#     "get_J2",
#     "get_J2_stack",
#     "get_KE",
#     "get_KE_stack",
#     "get_phi_from_bulk_density",
#     "get_phi_from_bulk_density_stack",
#     "get_phi_from_L",
#     "get_plastic_strain",
#     "get_plastic_strain_stack",
#     "get_pressure",
#     "get_pressure_stack",
#     "get_q_vm",
#     "get_q_vm_stack",
#     "get_scalar_shear_strain",
#     "get_scalar_shear_strain_stack",
#     "get_scalar_shear_stress",
#     "get_scalar_shear_stress_stack",
#     "get_skew_tensor",
#     "get_skew_tensor_stack",
#     "get_small_strain",
#     "get_small_strain_stack",
#     "get_strain_rate_from_L",
#     "get_strain_rate_from_L_stack",
#     "get_sym_tensor",
#     "get_sym_tensor_stack",
#     "get_volumetric_strain",
#     "get_volumetric_strain_stack",
#     "phi_to_e",
#     "phi_to_e_stack",
#     "plot_simple",
#     "PlotHelper",
#     "make_plots",
# ]
