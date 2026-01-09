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

from .common.simstate import SimState

from .common.rerun import RerunVisualizer
from .common.vtk_io import VTKVisualizer
from .common.sim_io import SimIO

from .grid.grid import GridState

from .shapefunctions.mapping import InteractionCache, ShapeFunctionMapping

from .constitutive_laws.constitutive_law import (
    ConstitutiveLawState,
    ConstitutiveLaw,
)

from .constitutive_laws.newtonfluid import NewtonFluid, NewtonFluidState
from .constitutive_laws.mu_i_rheology import MuI_LC, MuIState

from .constitutive_laws.linearelastic import LinearElasticLaw, LinearElasticState

from .solvers.usl import USLSolver, USLSolverState

from .solvers.usl_asflip import USLAFLIPState, USLAFLIP

from .forces.force import Force, BaseForceState

from .forces.gravity import Gravity, GravityState

from .forces.gridcontact import GridContact


from .solvers.coupling import BodyCoupling

from .constitutive_laws.modifiedcamclay import ModifiedCamClay, ModifiedCamClayState

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
)

from .forces.sdf_collider import SDFCollider
from .forces.boundary import PlanarBoundaries
from .utils.generate_body import generate_particles_in_sdf


from .common.builder import SimBuilder


hook.uninstall()
del hook
