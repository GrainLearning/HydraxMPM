import os
from typing import Dict, Optional, Tuple

import equinox as eqx

from ..utils.jax_helpers import set_default_gpu


class Config(eqx.Module):
    store_every: Optional[int] = eqx.field(default=0, static=True)
    ppc: Optional[int] = eqx.field(default=1, static=True)
    file: Optional[str] = eqx.field(default="", static=True)
    project: Optional[str] = eqx.field(default="", static=True)
    dt: Optional[float] = eqx.field(default=0.0, static=True)
    num_steps: Optional[int] = eqx.field(default=0, static=True)
    total_time: float = eqx.field(default="", static=True)
    default_gpu_id: Optional[int] = eqx.field(default=-1, static=True)
    output_path: str = eqx.field(default="", static=True)
    shapefunction: Optional[str] = eqx.field(default="cubic", static=True)
    dim: int = eqx.field(static=True)  # config

    output: Dict | Tuple[str, ...] = eqx.field(static=True)

    # internal variables
    _padding: tuple = eqx.field(init=False, static=True, repr=False)

    def __init__(
        self,
        total_time: Optional[float] = None,
        num_steps: Optional[int] = None,
        dt: Optional[float] = None,
        store_every: Optional[int] = 0,
        file: Optional[str] = "",
        project: Optional[str] = "",
        default_gpu_id: Optional[int] = -1,
        output_path: Optional[str] = None,
        shapefunction: Optional[str] = "cubic",
        ppc: Optional[int] = 1,
        dim: Optional[int] = 3,
        output: Optional[dict | Tuple[str, ...]] = None,
        **kwargs,
    ):
        if output is None:
            self.output = dict()
        else:
            self.output = output

        if total_time and num_steps:
            dt = total_time / num_steps
        elif total_time and dt:
            num_steps = int(total_time / dt)
        elif num_steps and dt:
            total_time = num_steps * dt
        else:
            raise ValueError("Please provide either total_time and num_steps or dt.")

        self.total_time = total_time
        self.num_steps = num_steps
        self.store_every = store_every
        self.dt = dt
        self.dim = dim

        # internal use...
        self._padding = (0, 3 - self.dim)

        if output_path is not None:
            self.output_path = output_path
        else:
            if file is not None:
                self.output_path = os.path.dirname(file) + "/output/" + project + "/"
            else:
                self.output_path = "/output/" + project + "/"

        self.project = project
        self.ppc = ppc

        # Particle-node connectivity, shape functions
        self.shapefunction = shapefunction
        if default_gpu_id != -1:
            set_default_gpu(self.default_gpu_id)
