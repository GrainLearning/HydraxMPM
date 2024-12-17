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

    pressure_stack = particles.get("pressure_stack")

    np.testing.assert_allclose(pressure_stack, jnp.array([0.0, 0.0]))

    # initialize with constant pressure
    particles = hdx.Particles(
        config=config,
        position_stack=jnp.zeros((config.num_points, 2)),
        pressure_ref=100.0,
    )

    pressure_stack = particles.get("pressure_stack")

    np.testing.assert_allclose(pressure_stack, jnp.array([100.0, 100.0]))

    # initialize with array of pressures
    particles = hdx.Particles(
        config=config,
        position_stack=jnp.zeros((config.num_points, 2)),
        pressure_ref=jnp.array([50.0, 100]),
    )

    pressure_stack = particles.get("pressure_stack")

    np.testing.assert_allclose(pressure_stack, jnp.array([50.0, 100.0]))

    # initialize with scalar of density
    particles = hdx.Particles(
        config=config, position_stack=jnp.zeros((config.num_points, 2)), density_ref=1.0
    )

    np.testing.assert_allclose(particles.mass_stack, jnp.array([0.01, 0.01]))

    # initialize with array of density
    particles = hdx.Particles(
        config=config,
        position_stack=jnp.zeros((config.num_points, 2)),
        density_ref=jnp.ones(2),
    )

    np.testing.assert_allclose(particles.mass_stack, jnp.array([0.01, 0.01]))

    print(particles.mass_stack)


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
