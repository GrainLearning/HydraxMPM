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
import equinox as eqx


from ..grid.grid import GridState
from ..material_points.material_points import MaterialPointState
from ..solvers.coupling import BodyCoupling

from typing import Self, List, Optional, Tuple, Dict

class BaseForceState(eqx.Module):
    pass


from ..shapefunctions.mapping import InteractionCache
from ..sdf.sdfobject import SDFObjectState

class Force(eqx.Module):

    def apply_kinematics(
        self: Self,
        mp_states: List[MaterialPointState],
        grid_states: List[GridState],
        f_states: List[Optional[BaseForceState | SDFObjectState]],
        intr_caches: Dict[Tuple[int, int], InteractionCache],
        couplings: Tuple[BodyCoupling,...],
        dt,
        time,
    ):
        """
        Apply hook 1 after interactions and connectivity is computed within the solver
        (e.g., Move rigid body)
        """
        return f_states

    def apply_pre_p2g(
        self: Self,
        mp_states: List[MaterialPointState],
        grid_states: List[GridState],
        f_states: List[Optional[BaseForceState | SDFObjectState]],
        intr_caches: Dict[Tuple[int, int], InteractionCache],
        couplings: Tuple[BodyCoupling,...],
        dt,
        time,
    ):
        """
        Apply hook 2 nefore particle to grid transfer, acting on particles.
        """
        return mp_states, f_states

    def apply_grid_forces(
        self: Self,
        mp_states: List[MaterialPointState],
        grid_states: List[GridState],
        f_states: List[Optional[BaseForceState | SDFObjectState]],
        intr_caches: Dict[Tuple[int, int], InteractionCache],
        couplings: Tuple[BodyCoupling,...],
        dt,
        time,
    ):
        """
        Apply hook 3 before integration acting on grid.

        Modify grid forces here directly.

        """
        return grid_states, f_states

    def apply_grid_moments(
        self: Self,
        mp_states: List[MaterialPointState],
        grid_states: List[GridState],
        f_states: List[Optional[BaseForceState | SDFObjectState]],
        intr_caches: Dict[Tuple[int, int], InteractionCache],
        couplings: Tuple[BodyCoupling,...],
        dt,
        time,
    ):
        """
        Apply hook 4 before g2p (after integration) acting on grid.

        The grid momentum can be modified here.

        """
        return grid_states, f_states
