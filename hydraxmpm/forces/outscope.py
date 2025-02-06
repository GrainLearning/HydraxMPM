"""Module for the gravity force. Impose gravity on the nodes."""

from typing import Tuple
import jax
import chex
import equinox as eqx
import jax.numpy as jnp
from typing_extensions import Self

from ..config.mpm_config import MPMConfig
from ..nodes.nodes import Nodes
from ..particles.particles import Particles
from .force import Forces


def _numpy_tuple(x: jnp.ndarray) -> tuple:
    assert x.ndim == 1
    return tuple([sub_x.item() for sub_x in x])


class OutScope(Forces):
    thickness: jnp.float32
    thickness: float = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))

    def __init__(self: Self, config: MPMConfig, thickness=3) -> Self:
        self.thickness = config.cell_size * thickness * jnp.ones(config.dim)
        super().__init__(config)

    def apply_on_particles(
        self: Self,
        particles: Particles = None,
        nodes: Nodes = None,
        step: int = 0,
    ) -> Tuple[Particles, Self]:
        def check_in_domain(pos):
            ls_valid = pos > jnp.array(self.config.origin) + jnp.array(self.thickness)
            gt_valid = pos < jnp.array(self.config.end) - jnp.array(self.thickness)
            return jnp.all(ls_valid * gt_valid)

        is_valid_stack = jax.vmap(check_in_domain)(particles.position_stack)

        new_mass_stack = jnp.where(is_valid_stack, particles.mass_stack, 0.0)

        new_volume_stack = jnp.where(is_valid_stack, particles.volume_stack, 0.0)

        def clip_higher_order_arrays(is_valid, pos, L):
            new_pos = jax.lax.cond(
                is_valid, lambda: pos, lambda: jnp.array(self.config.origin)
            )
            new_L = jax.lax.cond(is_valid, lambda: L, lambda: jnp.zeros((3, 3)))
            return new_pos, new_L

        new_position_stack, new_L_stack = jax.vmap(clip_higher_order_arrays)(
            is_valid_stack, particles.position_stack, particles.L_stack
        )
        # jax.debug.print("{}", new_mass_stack.shape)
        # # new_volume_stack = particles.volume_stack.at[invalid_index_stack].set(0.0)
        # # new_position_stack = particles.position_stack.at[invalid_index_stack].set(
        # #     jnp.array(self.config.origin)
        # # )
        # # new_L_stack = particles.L_stack.at[invalid_index_stack].set(jnp.zeros((3, 3)))

        new_particles = eqx.tree_at(
            lambda state: (
                state.mass_stack,
                state.volume_stack,
                state.position_stack,
                state.L_stack,
            ),
            particles,
            (
                new_mass_stack,
                new_volume_stack,
                new_position_stack,
                new_L_stack,
            ),
        )

        return new_particles, self
        # return particles, self
