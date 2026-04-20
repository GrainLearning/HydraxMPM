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

from .solvers.solver import BaseSolverState

from .material_points.material_points import (
    BaseMaterialPointState,
    MaterialPointState,
)

from .common.simstate import WorldState, MechanicsState, SimState

try:
    from .common.rerun import RerunVisualizer
except ImportError:
    RerunVisualizer = None

try:
    from .common.vtk_io import VTKVisualizer
except ImportError:
    VTKVisualizer = None

from .common.sim_io import SimIO

from .grid.grid import GridDomain

from .shapefunctions.mapping import InteractionCache, ShapeFunctionMapping

from .constitutive_laws.constitutive_law import (
    ConstitutiveLawState,
    ConstitutiveLaw,
)

from .constitutive_laws.newtonfluid import NewtonFluid, NewtonFluidState
from .constitutive_laws.mu_i_rheology import MuI_LC, MuIState
from .constitutive_laws.modifiedcamclay import ModifiedCamClay, ModifiedCamClayState
from .constitutive_laws.linearelastic import LinearElasticLaw, LinearElasticState
from .constitutive_laws.druckerprager import DruckerPrager, DruckerPragerState


from .solvers.usl import USLSolver, USLSolverState

from .solvers.usl_asflip import USLAFLIPState, USLAFLIP

from .forces.force import Force, BaseForceState

from .forces.gravity import Gravity, GravityState

from .forces.damping import Damping, DampingState

from .forces.gridcontact import GridContact


from .solvers.coupling import BodyCoupling


from .sdf.sdfobject import SDFObjectBase, SDFObjectState
from .sdf.sdfcollection import (
    BoxSDF,
    SphereSDF,
    PlaneSDF,
    CapsuleSDF,
    CylinderSDF,
    TorusSDF,
    StarSDF,
    HollowCylinderSDF,
    DomainSDF,
    MaterialPointCloudSDF,
)

from .sdf.gridsdf import GridSDF

from .forces.sdf_collider import SDFCollider
from .utils.generate_body import generate_particles_in_sdf


from .element_tests.driver import ElementTestDriver

from .element_tests.triaxial_test import TriaxialTest

from .common.builder import SimBuilder

from .utils.math_helpers import (
    safe_norm,
    get_volumetric_strain_stack,
    quaternion_rotate,
    precondition_from_lithostatic,
    reconstruct_stress_from_triaxial,
    get_sym_tensor,
    get_spin_tensor,
    get_jaumann_increment,
    get_pressure,
    get_dev_stress,
    get_dev_strain,
    get_volumetric_strain,
    get_q_vm,
    inv_2x2_robust,
    rotation_2d
)

from jaxtyping import Array, Float


hook.uninstall()
del hook
