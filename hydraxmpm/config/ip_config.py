import os

import equinox as eqx
import jax
from typing_extensions import Self, Optional
import dataclasses


class IPConfig(eqx.Module):
    # solver
    dt: float = eqx.field(static=True, converter=lambda x: float(x))
    num_steps: int = eqx.field(static=True, converter=lambda x: int(x))

    # solver
    store_every: int = eqx.field(static=True, converter=lambda x: int(x))

    # solver
    total_time: float = eqx.field(static=True, converter=lambda x: int(x))

    # functions
    dir_path: str = eqx.field(static=True, init=False, converter=lambda x: str(x))

    # functions
    file: Optional[str] = eqx.field(default="", static=True, converter=lambda x: str(x))

    # functions
    project: Optional[str] = eqx.field(
        default="", static=True, converter=lambda x: str(x)
    )
    # Only dim=3 is supported
    # note this is used by the solver and materials
    dim: int = eqx.field(default=3, static=True, init=False)

    def __init__(
        self: Self,
        dt: Optional[float] = None,
        num_steps: Optional[int] = None,
        total_time: Optional[float] = None,
        store_every: Optional[int] = 1,
        cpu: Optional[bool] = True,
        project: Optional[str] = "",
        file: Optional[str] = "",
    ) -> Self:
        if all(x is None for x in [dt, num_steps, total_time]):
            raise ValueError("Please provide either dt, num_steps or total_time")

        if total_time and num_steps:
            dt = total_time / num_steps
        elif total_time and dt:
            num_steps = int(total_time / dt)
        elif num_steps and dt:
            total_time = num_steps * dt

        self.total_time = total_time
        self.num_steps = num_steps
        self.store_every = store_every
        self.dt = dt

        self.dir_path = os.path.dirname(file) + "/"
        self.file = file
        self.project = project

        if cpu:
            jax.config.update("jax_platform_name", "cpu")

        self.dim = 3
        # if is64bit:
        #     jax.config.update("jax_enable_x64", True)
