import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create():
    # volume control shear
    et_benchmark = hdx.VolumeControlShear(x_range=(0.0, 1.0), y_range=(0.0, 0.0))

    et_benchmark = et_benchmark.init_steps(num_steps=2)

    # check velocity gradient
    expected_L = np.array(
        [
            [[-0.0, 0.0, 0.0], [0.0, -0.0, 0.0], [0.0, 0.0, -0.0]],
            [[-0.0, 1.0, 0.0], [1.0, -0.0, 0.0], [0.0, 0.0, -0.0]],
        ]
    )
    np.testing.assert_allclose(et_benchmark.L_control_stack, expected_L)
