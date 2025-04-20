# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-


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

from .sip_benchmarks.sip_benchmarks import (
    TriaxialConsolidatedUndrained,
    TriaxialConsolidatedDrained,
    ConstantPressureShear,
    IsotropicCompression,
)

from .forces.boundary import Boundary
from .forces.slipstickboundary import SlipStickBoundary
from .forces.rigidparticles import RigidParticles
from .forces.force import Force
from .forces.gravity import Gravity
from .grid.grid import Grid
from .material_points.material_points import MaterialPoints
from .solvers.sip_solver import SIPSolver
from .solvers.mpm_solver import MPMSolver

from .solvers.usl import USL
from .solvers.usl_apic import USL_APIC
from .solvers.usl_asflip import USL_ASFLIP


from .utils.mpm_callback_helpers import npz_to_vtk
from .utils.plot import make_plot


from .plotting import helpers, viewer

from .utils.math_helpers import (
    get_sym_tensor_stack,
    get_pressure,
    get_pressure_stack,
    get_dev_stress,
    get_volumetric_strain,
    get_dev_strain,
    get_q_vm,
    get_q_vm_stack,
    get_dev_strain,
    get_scalar_shear_strain,
    get_inertial_number,
    get_inertial_number_stack
)

from .common.types import TypeFloat, TypeFloatMatrix3x3, TypeInt, TypeFloatScalarPStack

hook.uninstall()
del hook
