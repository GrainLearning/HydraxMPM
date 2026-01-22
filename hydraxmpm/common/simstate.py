
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


from typing import Tuple,Optional, Dict, List

from ..material_points.material_points import (
    BaseMaterialPointState
)

from ..solvers.solver import BaseSolverState


from ..constitutive_laws.constitutive_law import ConstitutiveLawState

from ..forces.force import BaseForceState, Force

from jaxtyping import Array, Float, Int

from ..solvers.coupling import BodyCoupling

from ..constitutive_laws.constitutive_law import ConstitutiveLaw
from ..sdf.sdfobject import SDFObjectState
from ..shapefunctions.mapping import InteractionCache
from ..grid.grid import GridArrays


class WorldState(eqx.Module):
    """ Sub-container for World State.

    This is physical entities in the world such as material points, grids, and SDF objects.

    Attributes:
        material_points: Tuple of material point states (usually one per material),
            can be standard material points or rigid bodies
        grids: Tuple of grid states
        sdfs: Tuple of SDF object states (usually one per SDF collider)
    
    """
    material_points: Tuple[BaseMaterialPointState, ...] = ()
    sdfs: Tuple[Optional[SDFObjectState], ...] = ()


class MechanicsState(eqx.Module):
    """State related to the solving process and interactions.
    
    Attributes:
        constitutive_laws: Tuple of constitutive law states (usually one per material)
        interactions: Dictionary of interaction caches between material points and grids ( keys (mp_idx, grid_idx))
        solvers: Tuple of solver states or numeric states (e.g., APIC affine matrices, usually one per material)
        forces: Tuple of force states (e.g., gravity)

    """
    # Constitutive law history variables (e.g. plastic strain)
    constitutive_laws: Tuple[Optional[ConstitutiveLawState], ...] = ()
    
    # solver internal states (e.g. AFLIP affine matrices)
    solvers: Tuple[Optional[BaseSolverState], ...] = ()
    
    # Force states (e.g. time-dependent gravity)
    forces: Tuple[Optional[BaseForceState], ...] = ()

class SimState(eqx.Module):
    """Top Level Container for Simulation state
    
    This contains all relevant simulation states and is updated within the main simulation loop.
    It can be sharded easily to introduce multi-GPU or distributed computing in future.

    Attributes:
        time: Global simulation time
        step: Current simulation step
        dt: Time step size
        world: WorldState containing material points, grids, and SDF objects
        mechanics: MechanicsState containing constitutive laws, interactions, solvers, and forces
    """

        
    world: WorldState
    mechanics: MechanicsState

    # Global
    time: Float[Array, ""] | float = 0.0
    step: Int[Array, ""] | int = 0
    dt:   Float[Array, ""] | float = 0.0



class ParticleGeometry(eqx.Module):
    """Transient geometry for particles relative to SDFs."""

    dists: Float[Array, "num_points"]
    normals: Float[Array, "num_points dim"]
    wall_vels: Float[Array, "num_points dim"]


class NodeGeometry(eqx.Module):
    """Transient geometry for grid nodes relative to SDFs."""

    dists: Float[Array, "num_nodes"]
    normals: Float[Array, "num_nodes dim"]
    wall_vels: Float[Array, "num_nodes dim"]
    friction: Float[Array, "num_nodes"]


class SimCache(eqx.Module):
    # Interaction caches (for e.g., P2G, G2P mappings)
    interactions: Dict[Tuple[int, int], InteractionCache]

    grids: List[GridArrays]

    # Geometry - Computed in Connectivity, used in Collider & G2P
    # Key: (p_idx, sdf_idx)
    mp_geoms: Dict[Tuple[int, int], ParticleGeometry]

    # keys are (g_idx, sdf_idx)
    node_geoms: Dict[Tuple[int, int], NodeGeometry]
