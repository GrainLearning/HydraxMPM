import os
import sys

import equinox as eqx
import jax
from typing_extensions import Generic, Self


class IPConfig(eqx.Module):
    num_points: int = eqx.field(static=True, converter=lambda x: int(x))
    dt: float = eqx.field(static=True, converter=lambda x: float(x))
    num_steps: int = eqx.field(static=True, converter=lambda x: int(x))
    store_every: int = eqx.field(static=True, converter=lambda x: int(x))
    dim: int = eqx.field(static=True, converter=lambda x: int(x))
    total_time: int = eqx.field(static=True, converter=lambda x: int(x))
    dir_path: str = eqx.field(static=True, converter=lambda x: str(x))
    project: str = eqx.field(static=True, converter=lambda x: str(x))

    def __init__(
        self: Self,
        dt=0.0,
        num_steps=0,
        store_every=0,
        num_points: int = 1,
        dim: int = 3,
        cpu: bool = True,
        project: str = "",
        **kwargs: Generic,
    ):
        # total_time: jnp.float32
        self.dim = dim

        self.num_steps = num_steps
        self.store_every = store_every
        self.dt = dt

        self.num_points = num_points

        self.total_time = dt * num_steps

        if "file" in kwargs:
            file = kwargs.get("file")
        else:
            file = sys.argv[0]

        self.dir_path = os.path.dirname(file) + "/"

        self.project = project

        if cpu:
            jax.config.update("jax_platform_name", "cpu")
            jax.config.update("jax_enable_x64", True)
