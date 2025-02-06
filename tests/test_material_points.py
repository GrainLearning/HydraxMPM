# """Unit tests for the MaterialPoints dataclass."""

import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    """Unit test to initialize material_points  2 material_points."""
    material_points = hdx.MaterialPoints(position_stack=jnp.zeros((2, 2)))
    assert isinstance(material_points, hdx.MaterialPoints)

    assert material_points.dim == 2
    assert material_points.num_points == 2


def test_post_init():
    material_points = hdx.MaterialPoints(
        position_stack=jnp.zeros((2, 2)),
        rho_0=1.0,
    )

    material_points = material_points.init_volume_from_cellsize(cell_size=0.1, ppc=2)

    print(material_points.volume_stack, material_points.volume0_stack)
    np.testing.assert_allclose(material_points.volume_stack, jnp.array([0.005, 0.005]))
    np.testing.assert_allclose(
        material_points.volume_stack, material_points.volume0_stack
    )

    # check density ref

    material_points = hdx.MaterialPoints(
        position_stack=jnp.zeros((2, 2)),
        rho_0=1400,
    )


def test_refresh():
    """Unit test to refresh the state of the material_points."""
    position_stack = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    material_points = hdx.MaterialPoints(
        position_stack=position_stack, L_stack=jnp.ones((2, 3, 3))
    )

    material_points = material_points._refresh()

    np.testing.assert_allclose(material_points.L_stack, jnp.zeros((2, 3, 3)))
