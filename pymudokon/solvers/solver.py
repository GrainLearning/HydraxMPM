"""Base solve module."""

import operator
from functools import partial
from typing import Callable, List, Tuple

import chex
import jax
import jax.numpy as jnp
from typing_extensions import Self

from ..forces.forces import Forces
from ..materials.material import Material
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from ..shapefunctions.shapefunctions import ShapeFunction


def scan(f, init, xs=None, reverse=False, unroll=1, store_every=1):
    """https://github.com/google/jax/discussions/12157"""
    store_every = operator.index(store_every)
    assert store_every > 0

    kwds = dict(reverse=reverse, unroll=unroll)

    if store_every == 1:
        return jax.lax.scan(f, init, xs=xs, **kwds)

    N, rem = divmod(len(xs), store_every)

    if rem:
        raise ValueError("store_every must evenly divide len(xs)")

    xs = xs.reshape(N, store_every, *xs.shape[1:])

    def f_outer(carry, xs):
        carry, ys = jax.lax.scan(f, carry, xs=xs, **kwds)
        jax.debug.print("step {} \r", xs[-1])
        return carry, [yss[-1] for yss in ys]

    return jax.lax.scan(f_outer, init, xs=xs, **kwds)


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10, 11))
def run_solver(
    solver,
    particles: Particles,
    nodes: Nodes,
    shapefunctions: ShapeFunction,
    material_stack: List[Material],
    forces_stack: List[Forces] = None,
    num_steps: jnp.int32 = 1,
    store_every: jnp.int32 = 1,
    particles_keys: Tuple[str] = None,
    nodes_keys: Tuple[str] = None,
    materials_keys: Tuple[str] = None,
    forces_keys: Tuple[str] = None,
):
    if forces_stack is None:
        forces_stack = []

    if particles_keys is None:
        particles_keys = ()

    if nodes_keys is None:
        nodes_keys = ()

    if materials_keys is None:
        materials_keys = ()

    if forces_keys is None:
        forces_keys = ()

    def scan_fn(carry, control):
        step, solver, particles, nodes, shapefunctions, material_stack, forces_stack = carry

        solver, particles, nodes, shapefunctions, material_stack, forces_stack = solver.update_experimental(
            particles, nodes, shapefunctions, material_stack, forces_stack
        )

        carry = (step, solver, particles, nodes, shapefunctions, material_stack, forces_stack)

        accumulate = []

        for key in particles_keys:
            accumulate.append(particles.get(key))

        for key in nodes_keys:
            accumulate.append(nodes.get(key))

        for key in materials_keys:
            for material in material_stack:
                if key in material:
                    accumulate.append(material.get(key))

        for key in forces_keys:
            for force in forces_stack:
                if key in force:
                    accumulate.append(force.get(key))

        return carry, accumulate

    xs = jnp.arange(num_steps)

    return scan(
        scan_fn,
        (0, solver, particles, nodes, shapefunctions, material_stack, forces_stack),
        xs=xs,
        store_every=store_every,
        unroll=1,
    )
