"""Module to run solver and store its state."""

from functools import partial
from typing import Callable, List, Tuple

import chex
import jax
import jax.experimental
import jax.numpy as jnp

from ..forces.forces import Forces
from ..materials.material import Material
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction
from ..utils.jax_helpers import scan_kth
from .solver import Solver


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11))
def run_solver(
    solver: Solver,
    particles: Particles,
    nodes: Nodes,
    shapefunctions: ShapeFunction,
    material_stack: List[Material],
    forces_stack: List[Forces] = None,
    num_steps: jnp.int32 = 1,
    store_every: jnp.int32 = 1,
    particles_output: Tuple[str] = None,
    nodes_output: Tuple[str] = None,
    materials_output: Tuple[str] = None,
    forces_output: Tuple[str] = None,
) -> Tuple[
    Tuple[Particles, Nodes, ShapeFunction, List[Material], List[Forces]],
    Tuple[Solver, chex.Array],
]:
    """Run a MPM solver and store its state.

    Args:
        solver: Any solver class e.g., USL, USL_APIC
        particles: MPM particles dataclass
        nodes: Nodes dataclass
        shapefunctions: Shapefunctions dataclass
            e.g.,`LinearShapeFunction`, `CubicShapeFunction`
        material_stack:
            List of material dataclasses e.g., `LinearIsotropicElastic`
        forces_stack:
            List of forces. Defaults to None.
        num_steps: Total number of steps to run. Defaults to 1.
        store_every:
            Store data every nth step. Defaults to 1.
        particles_output: Properties entries to output of
            the particles e.g., `position_stack, velocity_stack`.
            Defaults to None.
        nodes_output: Node entries to output e.g., `masses`.
            Defaults to None.
        materials_output: Material properties to output.
            e.g., `eps_e_stack`. Defaults to None.
        forces_output: Force properties to output.
            Defaults to None.

    Returns:
        Tuple: Updated state, and output data.
    """
    if forces_stack is None:
        forces_stack = []

    if particles_output is None:
        particles_output = ()

    if nodes_output is None:
        nodes_output = ()

    if materials_output is None:
        materials_output = ()

    if forces_output is None:
        forces_output = ()

    def scan_fn(carry, control):
        step, solver, particles, nodes, shapefunctions, material_stack, forces_stack = (
            carry
        )

        solver, particles, nodes, shapefunctions, material_stack, forces_stack = (
            solver.update(
                particles, nodes, shapefunctions, material_stack, forces_stack
            )
        )
        # jax.debug.print("step {} ",step)

        carry = (
            step+1,
            solver,
            particles,
            nodes,
            shapefunctions,
            material_stack,
            forces_stack,
        )

        accumulate = []

        for key in particles_output:
            accumulate.append(particles.get(key))

        for key in nodes_output:
            accumulate.append(nodes.get(key))

        for key in materials_output:
            for material in material_stack:
                if key in material:
                    accumulate.append(material.get(key))

        for key in forces_output:
            for force in forces_stack:
                if key in force:
                    accumulate.append(force.get(key))

        return carry, accumulate

    xs = jnp.arange(num_steps)

    return scan_kth(
        scan_fn,
        (0, solver, particles, nodes, shapefunctions, material_stack, forces_stack),
        xs=xs,
        store_every=store_every,
        unroll=1,
    )

@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11))
def run_solver_io(
    solver: Solver,
    particles: Particles,
    nodes: Nodes,
    shapefunctions: ShapeFunction,
    material_stack: List[Material],
    forces_stack: List[Forces] = None,
    num_steps: jnp.int32 = 1,
    store_every: jnp.int32 = 1,
    callback: Callable = None,
    particles_output: Tuple[str] = None,
    nodes_output: Tuple[str] = None,
    materials_output: Tuple[str] = None,
    forces_output: Tuple[str] = None,
) -> Tuple[
    Tuple[Particles, Nodes, ShapeFunction, List[Material], List[Forces]],
    Tuple[Solver, chex.Array],
]:
        
    def main_loop(step,carry):
        solver, particles, nodes, shapefunctions, material_stack, forces_stack = (
            carry
        )

        solver, particles, nodes, shapefunctions, material_stack, forces_stack = (
            solver.update(
                particles, nodes, shapefunctions, material_stack, forces_stack, step
            )
        )

        carry = (
            solver,
            particles,
            nodes,
            shapefunctions,
            material_stack,
            forces_stack
        )
        
        return carry
    
    def scan_fn(carry, step):
        step_next = step + store_every
        solver, particles, nodes, shapefunctions, material_stack, forces_stack = jax.lax.fori_loop(
            step,
            step_next,
            main_loop,
            carry
        )
        
        carry = (
            solver,
            particles,
            nodes,
            shapefunctions,
            material_stack,
            forces_stack,
        )

        if callback:
            jax.debug.callback(callback,carry,step_next)
        
        return carry,[]
    
    xs = jnp.arange(0,num_steps,store_every).astype(jnp.int32)
    
    carry,accumulate = jax.lax.scan(
        scan_fn,
        (solver, particles, nodes, shapefunctions, material_stack, forces_stack),
        xs= xs,
        unroll=1
    )
    
    return carry