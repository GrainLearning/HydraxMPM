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

    ppc: Optional[int] = eqx.field(default=1, static=True)  # mpm solver

    project: Optional[str] = eqx.field(default="", static=True)  # basically output path

    output_path: str = eqx.field(default="", static=True)  # run sim

    shapefunction: Optional[str] = eqx.field(default="cubic", static=True)  # MPm solver

    dim: int = eqx.field(static=True)  # mpm

    output: Dict | Tuple[str, ...] = eqx.field(static=True)  # run sim

    # internal variables
    _padding: tuple = eqx.field(
        init=False, static=True, repr=False
    )  # internally for each module

    override_dir: bool = eqx.field(default=False, static=True)  # run sim
    create_dir: bool = eqx.field(default=False, static=True)  # run sim

    def __init__(
        self,
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
