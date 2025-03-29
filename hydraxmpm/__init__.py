"""HydraxMPM library.

Built with JAX.
"""

# ruff: noqa: E402
# ruff: noqa: F401
from jaxtyping import install_import_hook

hook = install_import_hook("hydraxmpm", "typeguard.typechecked")

from .constitutive_laws.constitutive_law import ConstitutiveLaw
from .common.base import Base
from .constitutive_laws.linearelastic import LinearIsotropicElastic
from .constitutive_laws.modifiedcamclay import ModifiedCamClay
from .constitutive_laws.mu_i_rheology_incompressible import MuI_incompressible
from .constitutive_laws.newtonfluid import NewtonFluid
from .constitutive_laws.druckerprager import DruckerPrager
from .et_benchmarks.et_benchmarks import TRX_CU, TRX_CD, S_CD, ISO_C

from .forces.boundary import Boundary
from .forces.rigidparticles import RigidParticles
from .forces.force import Force
from .forces.gravity import Gravity
from .grid.grid import Grid
from .material_points.material_points import MaterialPoints
from .solvers.config import Config
from .solvers.et_solver import ETSolver, run_et_solver
from .solvers.mpm_solver import MPMSolver
from .solvers.usl import USL
from .solvers.usl_apic import USL_APIC
from .solvers.usl_asflip import USL_ASFLIP
from .solvers.run_solvers import run_mpm


from .utils.mpm_callback_helpers import npz_to_vtk
from .utils.plot import make_plot, plot_set1, plot_set1_short


from .plotting import helpers, viewer

from .utils.math_helpers import (
    get_sym_tensor_stack,
    get_pressure,
    get_dev_stress,
    get_volumetric_strain,
    get_dev_strain,
    get_q_vm,
)

from .common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt, TypeFloatScalarPStack

hook.uninstall()
del hook
