# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM
"""
Explanation:
    This module contains the base `Force` logic class and a state
    which is used for consistent type checking and dependency injection.

    The force class contains hooks that are called at various stages of the MPM solver.
"""
from __future__ import annotations 

import equinox as eqx
from typing import Self, List, Optional, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    # fixes circular import with SimState
    from ..common.simstate import WorldState, MechanicsState

from ..grid.grid import GridState
from ..material_points.material_points import MaterialPointState
from ..solvers.coupling import BodyCoupling


from ..sdf.sdfobject import SDFObjectBase



class BaseForceState(eqx.Module):
    pass


from ..shapefunctions.mapping import InteractionCache
from ..sdf.sdfobject import SDFObjectState


class Force(eqx.Module):

    def apply_kinematics(
        self: Self,
        world: WorldState,
        mechanics: MechanicsState,
        sdf_logics: Tuple[SDFObjectBase,...],
        couplings: Tuple[BodyCoupling, ...],
        dt,
        time
    ) -> Tuple[WorldState, MechanicsState]:
        """
        Apply hook 1 (e.g., Move rigid body)
        """
        return world, mechanics

    def apply_pre_p2g(
        self,
        world: WorldState,
        mechanics: MechanicsState,
        sdf_logics: Tuple[SDFObjectBase, ...],
        couplings: Tuple[BodyCoupling, ...],
        dt,
        time
    ):
        """
        Apply hook 2 nefore particle to grid transfer, acting on particles.
        """
        return world, mechanics

    def apply_grid_forces(
        self: Self,
        world: WorldState,
        mechanics: MechanicsState,
        sdf_logics: Tuple[SDFObjectBase, ...],
        couplings: Tuple[BodyCoupling, ...],
        dt,
        time
    ):
        """
        Apply hook 3 before integration acting on grid.

        Modify grid forces here directly.

        """
        return world, mechanics

    def apply_grid_moments(
        self: Self,
        world: WorldState,
        mechanics: MechanicsState,
        sdf_logics: Tuple[SDFObjectBase, ...],
        couplings: Tuple[BodyCoupling, ...],
        dt,
        time
    ):
        """
        Apply hook 4 before g2p (after integration) acting on grid.

        The grid momentum can be modified here.

        """
        return world, mechanics
