# TODO Check in solver if timestep is set
# TODO Check if particles are set in solver

import dataclasses
import os
from pprint import pp

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, UInt
from typing_extensions import Optional, Self, Union

from ..utils.jax_helpers import set_default_gpu


def _numpy_tuple(x) -> tuple:
    def _convert(x: Union[UInt[Array, "dim"], Float[Array, "dim"]]) -> tuple:
        assert x.ndim == 1
        return tuple([sub_x.item() for sub_x in x])

    return _convert(jnp.array(x))


def _numpy_tuple_deep(x):
    def _covert(
        x: Union[UInt[Array, "connectivity dim"], Float[Array, "connectivity dim"]],
    ) -> tuple:
        return tuple(map(_numpy_tuple, x))

    return _covert(jnp.array(x))


class MPMConfig(eqx.Module):
    # init necessary
    origin: tuple = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))

    end: tuple = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))

    cell_size: float = eqx.field(static=True, converter=lambda x: float(x))

    # post init
    inv_cell_size: float = eqx.field(
        init=False, static=True, converter=lambda x: float(x)
    )
    grid_size: tuple = eqx.field(
        init=False, static=True, converter=lambda x: _numpy_tuple(x)
    )
    dim: int = eqx.field(static=True, init=False)
    num_cells: int = eqx.field(init=False, static=True, converter=lambda x: int(x))
    forward_window: tuple = eqx.field(
        repr=False, init=False, static=True, converter=lambda x: _numpy_tuple_deep(x)
    )

    backward_window: tuple = eqx.field(
        repr=False, init=False, static=True, converter=lambda x: _numpy_tuple_deep(x)
    )
    window_size: int = eqx.field(init=False, static=True, converter=lambda x: int(x))
    padding: tuple = eqx.field(init=False, static=True, repr=False)
    dir_path: str = eqx.field(init=False, static=True, converter=lambda x: str(x))

    # default (set later, maybe)
    num_points: Optional[int] = eqx.field(
        default=0, static=True, converter=lambda x: int(x)
    )
    store_every: Optional[int] = eqx.field(
        default=0, static=True, converter=lambda x: int(x)
    )
    shapefunction: Optional[str] = eqx.field(
        default="cubic", static=True, converter=lambda x: str(x)
    )
    file: Optional[str] = eqx.field(default="", static=True, converter=lambda x: str(x))
    project: Optional[str] = eqx.field(
        default="", static=True, converter=lambda x: str(x)
    )
    default_gpu_id: Optional[int] = eqx.field(
        default=-1, static=True, converter=lambda x: int(x)
    )
    dt: Optional[float] = eqx.field(
        default=0.0, static=True, converter=lambda x: float(x)
    )
    ppc: Optional[int] = eqx.field(default=1, static=True, converter=lambda x: int(x))
    num_steps: Optional[int] = eqx.field(
        default=0, static=True, converter=lambda x: int(x)
    )
    device: Optional[jax.Device] = eqx.field(default=None, static=True)

    def __init__(
        self,
        origin: Float[Array, "dim"] | tuple,
        end: Float[Array, "dim"] | tuple,
        cell_size: float,
        num_points: Optional[int] = 0,
        store_every: Optional[int] = 0,
        shapefunction: Optional[str] = "cubic",
        project: Optional[str] = "",
        file: Optional[str] = "",
        default_gpu_id: Optional[int] = -1,
        dt: Optional[float] = 0.0,
        ppc: Optional[int] = 1,
        num_steps: Optional[int] = 0,
        device: Optional[jax.Device] = None,
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
            origin: start point of the domain box
            end: end point of the domain box
            cell_size: cell size of the background grid
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
        self.origin = origin
        self.end = end
        self.cell_size = cell_size
        self.num_points = num_points
        self.store_every = store_every
        self.shapefunction = shapefunction
        self.file = file
        self.default_gpu_id = default_gpu_id
        self.dt = dt
        self.ppc = ppc
        self.num_steps = num_steps
        self.project = project
        self.device = device

        # post init
        self.inv_cell_size = 1.0 / self.cell_size

        self.grid_size = (
            (jnp.array(self.end) - jnp.array(self.origin)) / self.cell_size + 1
        ).astype(jnp.uint32)

        self.num_cells = np.prod(self.grid_size).astype(jnp.uint32)

        self.dim = len(self.grid_size)

        # create connectivity
        if self.shapefunction == "linear":
            window_1D = jnp.arange(2).astype(jnp.uint32)
        elif self.shapefunction == "cubic":
            window_1D = jnp.arange(4).astype(jnp.uint32) - 1

        if self.dim == 2:
            self.forward_window = jnp.array(
                jnp.meshgrid(window_1D, window_1D)
            ).T.reshape(-1, self.dim)
        elif self.dim == 3:
            self.forward_window = jnp.array(
                jnp.meshgrid(window_1D, window_1D, window_1D)
            ).T.reshape(-1, self.dim)
        elif self.dim == 1:
            self.forward_window = window_1D  # not tested!
            raise NotImplementedError

        self.backward_window = self.forward_window[::-1] - 1
        self.window_size = len(self.backward_window)

        self.padding = (0, 3 - self.dim)

        self.dir_path = os.path.dirname(self.file) + "/"

        if self.default_gpu_id != -1:
            set_default_gpu(self.default_gpu_id)

    def replace(self, **kwargs) -> Self:
        return dataclasses.replace(self, **kwargs)
