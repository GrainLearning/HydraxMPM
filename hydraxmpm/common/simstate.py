
# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-
"""
Explaination:
    This module contains the global SimState which hold all relevant simulation states.

"""


import equinox as eqx


from typing import Tuple,Optional, Dict

from ..material_points.material_points import (
    BaseMaterialPointState
)
from ..grid.grid import GridState

from ..shapefunctions.mapping import InteractionCache

from ..solvers.solver import BaseSolverState


from ..constitutive_laws.constitutive_law import ConstitutiveLawState

from ..forces.force import BaseForceState, Force

from jaxtyping import Array, Float, Int

from ..solvers.coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..sdf.sdfobject import SDFObjectState


class SimState(eqx.Module):
    """
    Sim state container.

    This contains all relevant simulation states and is updated within the main simulation loop.

    It can be sharded easily to introduce multi-GPU or distributed computing in future.

    Attributes:
        time: Global simulation time
        step: Current simulation step
        dt: Time step size
        grids: Tuple of grid states
        material_points: Tuple of material point states (usually one per material),
          can be standard material points or rigid bodies
        constitutive_laws: Tuple of constitutive law states (usually one per material)
        solvers: Tuple of solver states or numeric states (e.g., APIC affine matrices, usually one per material)
        interactions: Dictionary of interaction caches between material points and grids ( keys (mp_idx, grid_idx))
        forces: Tuple of force states, may also contain SDF object states
    """

    # Global time and step
    time: Float[Array, ""] | float = 0.0
    step: Int[Array, ""] | int = 0
    dt:   Float[Array, ""] | float = 0.0

    # Body
    grids: Tuple[Optional[GridState],...] =()
    material_points: Tuple[BaseMaterialPointState, ...] = ()
    constitutive_laws: Tuple[Optional[ConstitutiveLawState], ...] = ()
    interactions: Dict[Tuple[int, int], InteractionCache] = eqx.field(default_factory=dict)
    
    # numeric
    solvers: Tuple[Optional[BaseSolverState], ...] = ()
    
    # Pairwise couplings between material points and grids
    forces: Tuple[Optional[BaseForceState | SDFObjectState], ...] = ()

