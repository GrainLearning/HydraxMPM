# TODO Check in solver if timestep is set
# TODO Check if particles are set in solver

import dataclasses
import os
from pprint import pp

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int, UInt
from typing_extensions import Optional, Self, Union

from ..utils.jax_helpers import set_default_gpu


def _numpy_tuple(x) -> tuple:
    def _convert(x) -> tuple:
        assert x.ndim == 1
        return tuple([sub_x.item() for sub_x in x])

    return _convert(jnp.array(x))


class MPMConfig(eqx.Module):
    # init necessary

    _num_points: Optional[int] = eqx.field(
        default=0, static=True, converter=lambda x: int(x)
    )

    def __init__(
        self,
        _num_points: Optional[int] = 0,
    ):
        """Configuration object for MPM simulations.

        This dataclass stores parameters for setting up and running Material Point Method (MPM) simulations.

        It is immutable; use the `.replace()` method to create a modified copy.

        ```python
        import hydraxmpm as hdx
        import jax.numpy as jnp

        config = hdx.MPMConfig(
            origin=jnp.array([0.0, 0.0]),
            end=jnp.array([5.0, 5.0]),
            cell_size=0.1,
            file=__file__,  # Store location of the current file for output
            dt=1e-4, # Example timestep
        )

        # Modify time step
        dt_critical = 1e-5
        config = config.replace(dt=dt_critical)
        ```

        Args:

            num_points: number of particles. Defaults to 0.
            store_every: store every nth step. Defaults to 0.
            shapefunction: shapefunction type, either "linear" or "cubic". Defaults to "cubic".
            file: location of current file (e.g., use __file__).
                This is used to store output. Defaults to "".
            default_gpu_id: selects gpu id to run on. Defaults to -1.
            dt: time step. Defaults to 0.0.
            ppc: particles per cell in one dimension. Defaults to 1.
            num_steps: step count to run the simulation for. Defaults to 0.
        """

        # post init
        self._dim = len(origin)

        self._dir_path = os.path.dirname(self.file) + "/"

        if self.default_gpu_id != -1:
            set_default_gpu(self.default_gpu_id)

    def replace(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)
