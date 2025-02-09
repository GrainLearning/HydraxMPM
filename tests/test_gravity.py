import equinox as eqx
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to initialize gravity."""

    gravity = hdx.Gravity(gravity=jnp.array([0.0, 0.0, 0.9]))
    assert isinstance(gravity, hdx.Gravity)


def test_apply_grav_on_particles():
    material_points = hdx.MaterialPoints(
        position_stack=jnp.array([[0.0, 0.0], [0.5, 0.5]]),
        mass_stack=jnp.array([1.0, 1.0]),
    )

    grav = hdx.Gravity(gravity=jnp.array([0.0, -9.8]))

    new_particles, grav = grav.apply_on_points(material_points, step=1)

    force_stack = new_particles.force_stack

    np.testing.assert_allclose(force_stack, jnp.array([[0.0, -9.8], [0.0, -9.8]]))
