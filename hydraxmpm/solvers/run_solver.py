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
from ..utils.jax_helpers import scan_kth
from .solver import Solver
from .usl import USL

from ..config.mpm_config import MPMConfig


@partial(
    jax.jit,
    static_argnames=(
        "config",
        "particles_output",
        "nodes_output",
        "materials_output",
        "forces_output",
    ),
)
def run_solver(
    config: MPMConfig,
    solver: USL,
    particles: Particles,
    nodes: Nodes,
    material_stack: List[Material],
    forces_stack: List[Forces] = None,
    particles_output: Tuple[str] = None,
    nodes_output: Tuple[str] = None,
    materials_output: Tuple[str] = None,
    forces_output: Tuple[str] = None,
) -> Tuple[
    Tuple[Particles, Nodes, List[Material], List[Forces]],
    Tuple[USL, chex.Array],
]:
    """Run a MPM solver and store its state.

    Args:
        solver: Any solver class e.g., USL, USL_APIC
        particles: MPM particles dataclass
        nodes: Nodes dataclass
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
        (
            prev_step,
            prev_solver,
            prev_particles,
            prev_nodes,
            prev_material_stack,
            prev_forces_stack,
        ) = carry

        (
            new_solver,
            new_particles,
            new_nodes,
            new_material_stack,
            new_forces_stack,
        ) = prev_solver.update(
            prev_particles,
            prev_nodes,
            prev_material_stack,
            prev_forces_stack,
            prev_step,
        )

        new_step = prev_step + 1

        new_carry = (
            new_step,
            new_solver,
            new_particles,
            new_nodes,
            new_material_stack,
            new_forces_stack,
        )

        accumulate = []

        for key in particles_output:
            accumulate.append(new_particles.__getattribute__(key))

        for key in nodes_output:
            accumulate.append(new_nodes.__getattribute__(key))

        for key in materials_output:
            for new_material in new_material_stack:
                accumulate.append(new_material.__getattribute__(key))

        for key in forces_output:
            for new_force in new_forces_stack:
                if key in new_force.__dict__.keys():
                    accumulate.append(new_force.__getattribute__(key))

        return new_carry, accumulate

    xs = jnp.arange(config.num_steps)

    return scan_kth(
        scan_fn,
        (
            0,
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        ),
        xs=xs,
        store_every=config.store_every,
        unroll=False,
    )


@partial(
    jax.jit,
    static_argnames=("config", "callback"),
)
def run_solver_io(
    config: MPMConfig,
    solver: Solver,
    particles: Particles,
    nodes: Nodes,
    material_stack: List[Material],
    forces_stack: List[Forces] = None,
    callback: Callable = None,
) -> Tuple[
    Tuple[Particles, Nodes, List[Material], List[Forces]],
    Tuple[Solver, chex.Array],
]:
    def main_loop(step, carry):
        solver, particles, nodes, material_stack, forces_stack = carry

        solver, particles, nodes, material_stack, forces_stack = solver.update(
            particles, nodes, material_stack, forces_stack, step
        )

        carry = (solver, particles, nodes, material_stack, forces_stack)

        return carry

    def scan_fn(carry, step):
        step_next = step + config.store_every
        solver, particles, nodes, material_stack, forces_stack = (
            jax.lax.fori_loop(step, step_next, main_loop, carry)
        )

        carry = (
            solver,
            particles,
            nodes,
            material_stack,
            forces_stack,
        )

        callback(carry, step_next)

        return carry, []

    xs = jnp.arange(0, config.num_steps, config.store_every).astype(jnp.int32)

    carry, accumulate = jax.lax.scan(
        scan_fn,
        (solver, particles, nodes, material_stack, forces_stack),
        xs=xs,
        unroll=False,
    )

    return carry
