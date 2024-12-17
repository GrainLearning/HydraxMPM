import jax.numpy as jnp
import hydraxmpm as hdx


def test_create_replace():
    config = hdx.MPMConfig(
        origin=jnp.array([0.0, 0.0, 0.0]),
        end=jnp.array([1.0, 1.0, 1.0]),
        cell_size=2.0,
        ppc=2,
        dt=1e-3,
        num_steps=10000,
    )

    config = config.replace(file=__file__)
