import os
from typing import Dict, Optional, Tuple

import equinox as eqx

from ..utils.jax_helpers import set_default_gpu

from pathlib import Path

import shutil


import sys
from pathlib import Path


def get_script_directory() -> Path:
    """Get the directory of the script that imported this module."""
    main_module = sys.modules.get("__main__")
    if hasattr(main_module, "__file__"):
        script_path = Path(main_module.__file__).resolve()
        return script_path.parent
    # Fallback if __main__ lacks __file__ (e.g., interactive shell)
    return Path.cwd()


class Config(eqx.Module):
    """

    Configuration for the solvers:

    - [MPMSolver][solvers.mpm_solver.MPMSolver]
    - [ETSolver][solvers.mpm_solver.ETSolver]




    """

    store_every: Optional[int] = eqx.field(default=0, static=True)
    ppc: Optional[int] = eqx.field(default=1, static=True)
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

    override_dir: bool = eqx.field(default=False, static=True)
    create_dir: bool = eqx.field(default=False, static=True)

    def __init__(
        self,
        total_time: Optional[float] = None,
        num_steps: Optional[int] = None,
        dt: Optional[float] = None,
        store_every: Optional[int] = 0,
        project: Optional[str] = "",
        output_path: Optional[str] = None,
        shapefunction: Optional[str] = "cubic",
        ppc: Optional[int] = 1,
        dim: Optional[int] = 3,
        output: Optional[dict | Tuple[str, ...]] = None,
        create_dir: Optional[bool] = False,
        override_dir: Optional[bool] = False,
        **kwargs,
    ):
        """

        Note:
            At least two of the following must be defined total_time, num_steps,
            dt for the config to be valid



        Args:
            total_time: Total run time. Defaults to None.
            num_steps: Total number of steps. Defaults to None.
            dt: Time increment. Defaults to None.
            store_every: Store at every nth step. Defaults to 0.
            project: Output project location. Defaults to "".
            default_gpu_id: Default GPU ID. Defaults to -1.
            output_path: Output path. Defaults to None.
            shapefunction: Shape function type. Defaults to "cubic".
            ppc: Particles per cell. Defaults to 1.
            dim: Either 3D (`3`), or plain strain (`2`). Defaults to 3.
            output: A `Dict` of outputs if MPM solver is used,
                and a`List` of outputs if element test solver is used.
                See the [How to visualize results](../how-tos/visualize.md) for further details.
                Defaults to None.

        """
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

        # Example usage in your module

        if output_path is not None:
            self.output_path = output_path
        else:
            script_dir = get_script_directory()
            self.output_path = (script_dir / "output" / project).as_posix()

        self.override_dir = override_dir
        self.create_dir = create_dir

        if self.create_dir:
            dirpath = Path(self.output_path)
            if not dirpath.exists():
                dirpath.mkdir(parents=True)

            if self.override_dir:
                shutil.rmtree(dirpath)
                dirpath.mkdir(parents=True)

        self.project = project
        self.ppc = ppc

        self.shapefunction = shapefunction
