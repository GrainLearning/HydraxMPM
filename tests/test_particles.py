# """Unit tests for the Particles dataclass."""

import chex
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to initialize particles over 2 particles."""
    config = hdx.MPMConfig(
        origin=[0.0, 0.0],
        end=[
            1.0,
            1.0,
        ],
        cell_size=0.1,
        num_points=2,
    )

    particles = hdx.Particles(
        config=config,
        position_stack=jnp.zeros((config.num_points, 2)),
    )
    chex.assert_shape(particles.position_stack, (config.num_points, 2))


def test_calculate_volume():
    """Unit test to calculate the volume of the particles.

    Volume calculation is based on the background grid discretization.
    """
    position_stack = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=0.5,
        num_points=position_stack.shape[0],
        ppc=1,
    )
    particles = hdx.Particles(config=config, position_stack=position_stack)

    volume_stack = particles.calculate_volume()

    np.testing.assert_allclose(volume_stack, jnp.array([0.125, 0.125]))


def test_refresh():
    """Unit test to refresh the state of the particles."""
    position_stack = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    config = hdx.MPMConfig(
        origin=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        cell_size=0.5,
        num_points=position_stack.shape[0],
        ppc=1,
    )

    particles = hdx.Particles(
        config=config,
        position_stack=position_stack,
        L_stack=jnp.ones((2, 3, 3)),
    )

    particles = particles.refresh()

    np.testing.assert_allclose(particles.L_stack, jnp.zeros((2, 3, 3)))
