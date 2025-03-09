"""HydraxMPM library.

Built with JAX.
"""

# ruff: noqa: E402
# ruff: noqa: F401
from jaxtyping import install_import_hook

hook = install_import_hook("hydraxmpm", "typeguard.typechecked")

from .common.base import Base
from .constitutive_laws.linearelastic import LinearIsotropicElastic
from .constitutive_laws.modifiedcamclay import ModifiedCamClay
from .constitutive_laws.mu_i_rheology_incompressible import MuI_incompressible
from .constitutive_laws.exp_mu_i_phi_i import ExpMuIPhI
from .constitutive_laws.exp_mu_i_csl import ExpMuICSL
from .constitutive_laws.newtonfluid import NewtonFluid
from .constitutive_laws.exp_mu_i_phi_i_lc import ExpMuIPhiILC
from .et_benchmarks.et_benchmarks import (
    ConstantPressureSimpleShear,
    ConstantVolumeSimpleShear,
    ETBenchmark,
    IsotropicCompression,
    TriaxialCompressionDrained,
)
from .forces.boundary import Boundary
from .forces.force import Force
from .forces.gravity import Gravity
from .grid.grid import Grid
from .material_points.material_points import MaterialPoints
from .solvers.config import Config
from .solvers.et_solver import ETSolver
from .solvers.mpm_solver import MPMSolver
from .solvers.usl import USL
from .solvers.usl_apic import USL_APIC
from .solvers.usl_asflip import USL_ASFLIP
from .utils.mpm_callback_helpers import npz_to_vtk
# from .utils.plot import make_plot, plot_set1, plot_set1_short

hook.uninstall()

# from .experimental.mcc_nlncl import MCC_NLNCL
# from .experimental.mu_i_nlncl import MuI_NLNCL

# from .materials.druckerprager import DruckerPrager

# from .solvers.config import Config
# from .solvers.ip_solver import IPSolver

#
#     # from .plotting.plot_helper import PlotHelper
# from .utils.math_helpers import (
#     #     e_to_phi,
#     #     e_to_phi_stack,
#     #     get_dev_strain,
#     #     get_dev_strain_stack,
#     #     get_dev_stress,
#     #     get_dev_stress_stack,
#     #     get_e_from_bulk_density,
#     #     get_hencky_strain_stack,
#     #     get_inertial_number,
#     #     get_inertial_number_stack,
#     #     get_J2,
#     #     get_J2_stack,
#     #     get_k0_stress,
#     #     get_KE,
#     #     get_KE_stack,
#     #     get_phi_from_bulk_density,
#     #     get_phi_from_bulk_density_stack,
#     #     get_phi_from_L,
#     #     get_plastic_strain,
#     #     get_plastic_strain_stack,
#     get_pressure,
#     get_pressure_stack,
#     get_q_vm,
#     get_q_vm_stack,
#     #     get_scalar_shear_strain,
#     #     get_scalar_shear_strain_stack,
#     #     get_scalar_shear_stress,
#     #     get_scalar_shear_stress_stack,
#     #     get_skew_tensor,
#     #     get_skew_tensor_stack,
#     #     get_small_strain,
#     #     get_small_strain_stack,
#     #     get_strain_rate_from_L,
#     #     get_strain_rate_from_L_stack,
#     #     get_sym_tensor,
#     #     get_sym_tensor_stack,
#     #     get_volumetric_strain,
#     #     get_volumetric_strain_stack,
#     #     phi_to_e,
#     #     phi_to_e_stack,
# )


# # from .forces.rigidparticles import RigidParticles
# # from .forces.outscope import OutScope

# # from .materials.mu_i_rheology_incompressible import MuI_incompressible
# # from .forces.rigidparticles import RigidParticles
# # from .materials.modifiedcamclay import ModifiedCamClay
# # from .materials.pg.uh import UH
# # from .solvers.usl_asflip import USL_ASFLIP
# # from .utils.jax_helpers import dump_restart_files, get_dirpath, get_sv, set_default_gpu


# # from .utils.mpm_callback_helpers import (
# #     io_vtk_callback,
# #     io_material_point_callback,
# #     io_movie_callback,
# # )
# # from .utils.mpm_domain_helpers import (
# #     discretize,
# #     fill_domain_with_particles,
# #     generate_mesh,
# # )
# # from .utils.mpm_plot_helpers import (
# #     PvPointHelper,
# #     make_pvplots,
# #     point_to_3D,
# #     points_to_3D,
# # )
# # from .utils.stl_helpers import (
# #     get_stl_bounds,
# #     sample_points_in_volume,
# #     sample_points_on_surface,
# # )


# # # # from .utils.mpm_postprocessing_helpers import (
# # # #     post_processes_grid_gradient_stack,
# # # #     post_processes_stress_stack,
# # # # )


# # # # __all__ = [
# # # #     "Nodes",
# # # #     "Particles",
# # # #     "ShapeFunction",
# # # #     "LinearShapeFunction",
# # # #     "CubicShapeFunction",
# # # #     "DirichletBox",
# # # #     "RigidParticles",
# # # #     "Forces",
# # # #     "Gravity",
# # # #     "NodeWall",
# # # #     "Material",
# # # #     "LinearIsotropicElastic",
# # # #     "NewtonFluid",
# # # #     "DruckerPrager",
# # # #     "ModifiedCamClay",
# # # #     "MCC_MRM",
# # # #     "MuI",
# # # #     "mix_control",
# # # #     "MPBenchmark",
# # # #     "USL",
# # # #     "USL_APIC",
# # # #     "run_solver",
# # # #     "discretize",
# # # #     "e_to_phi",
# # # #     "e_to_phi_stack",
# # # #     "get_dev_strain",
# # # #     "get_dev_strain_stack",
# # # #     "get_dev_stress",
# # # #     "get_dev_stress_stack",
# # # #     "get_e_from_bulk_density",
# # # #     "get_inertial_number",
# # # #     "get_inertial_number_stack",
# # # #     "get_J2",
# # # #     "get_J2_stack",
# # # #     "get_KE",
# # # #     "get_KE_stack",
# # # #     "get_phi_from_bulk_density",
# # # #     "get_phi_from_bulk_density_stack",
# # # #     "get_phi_from_L",
# # # #     "get_plastic_strain",
# # # #     "get_plastic_strain_stack",
# # # #     "get_pressure",
# # # #     "get_pressure_stack",
# # # #     "get_q_vm",
# # # #     "get_q_vm_stack",
# # # #     "get_scalar_shear_strain",
# # # #     "get_scalar_shear_strain_stack",
# # # #     "get_scalar_shear_stress",
# # # #     "get_scalar_shear_stress_stack",
# # # #     "get_skew_tensor",
# # # #     "get_skew_tensor_stack",
# # # #     "get_small_strain",
# # # #     "get_small_strain_stack",
# # # #     "get_strain_rate_from_L",
# # # #     "get_strain_rate_from_L_stack",
# # # #     "get_sym_tensor",
# # # #     "get_sym_tensor_stack",
# # # #     "get_volumetric_strain",
# # # #     "get_volumetric_strain_stack",
# # # #     "phi_to_e",
# # # #     "phi_to_e_stack",
# # # #     "plot_simple",
# # # #     "PlotHelper",
# # # #     "make_plots",
# # # # ]
