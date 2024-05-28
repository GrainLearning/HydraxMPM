"""Base class for single integration point benchmark module"""


import jax.numpy as jnp
import xarray as xr
from jax import Array

from ..materials.material import Material


def simple_shear(
    material: Material,
    eps_path: Array,
    volumes: Array,
    dt: jnp.float32 | Array,
    results_to_store=[],
    output_step=1,
):
    """Perfoms a loading step the simple shear test on a material.

    Args:
    ----
        self (SimpleShearControl): self reference

    """
    num_benchmarks, num_steps = eps_path.shape

    eps_inc_target = jnp.zeros((num_benchmarks, 3, 3))
    eps_inc_prev = jnp.zeros((num_benchmarks, 3, 3))

    if (isinstance(dt, jnp.float64)) | (isinstance(dt, jnp.float32)) | (isinstance(dt, float)):
        dt = dt * jnp.ones(num_steps)

    datasets = []
    for step in range(num_steps):
        dt_step = dt[step]
        eps_inc_target = eps_inc_target.at[:, 0, 1].add(eps_path[:, step])
        strain_increment = eps_inc_target - eps_inc_prev

        strain_rate = strain_increment / dt_step

        stress, material = material.update_stress_benchmark(strain_rate, volumes, dt_step, update_history=True)

        eps_inc_prev = eps_inc_target.copy()

        if step % output_step == 0:
            print(f"Step {step} of {num_steps}")
            store = {
                "stress": (["benchmark", "i", "j"], stress.copy()),
                "strain_rate": (["benchmark", "i", "j"], strain_rate.copy()),
                "eps_path": (["benchmark", "i", "j"], eps_inc_target.copy()),
                "dt": (["benchmark"], jnp.array([dt_step]).repeat(num_benchmarks)),
            }
            for attr in results_to_store:
                if attr in material.__dict__:
                    value = material.__dict__[attr].copy()
                    if value.ndim != 3:
                        value = value.reshape((1,) * (3 - value.ndim) + value.shape)
                    store[attr] = (["benchmark", "i", "j"], value)

            data = xr.Dataset(store, coords={"step": step})
            datasets.append(data)

    datasets = xr.concat(datasets, dim="iter")
    return datasets, material
