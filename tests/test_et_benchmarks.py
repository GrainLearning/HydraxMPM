import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx


def test_create_volume_control_shear():
    # volume control shear
    et_benchmark = hdx.ConstantVolumeSimpleShear(x_range=(0.0, 1.0), y_range=(0.0, 0.0))

    material_points = hdx.MaterialPoints()
    config = hdx.Config(total_time=1.0, num_steps=2)
    et_benchmark, material_points = et_benchmark.init_state(config, material_points)


def test_create_pressure_control_shear():
    # volume control shear
    et_benchmark = hdx.ConstantPressureSimpleShear(
        x_range=(0.0, 1.0), p_range=(1000.0, 1000.0), init_material_points=True
    )

    material_points = hdx.MaterialPoints()
    config = hdx.Config(total_time=1.0, num_steps=2)
    et_benchmark, material_points = et_benchmark.init_state(config, material_points)

    # print(et_benchmark.X_control_stack)
    # print(material_points.p_stack)


# test_create_pressure_control_shear()
