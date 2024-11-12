from typing_extensions import Self, Generic

import equinox as eqx
import jax
import numpy as np
import os
import jax.numpy as jnp

from ..utils.jax_helpers import set_default_gpu

import sys


# _numpy_tuple = lambda arr: tuple([row for row in arr])
def _numpy_tuple(x: np.ndarray) -> tuple:
    # assert x.ndim == 1
    # return tuple([sub_x.item() for sub_x in x])
    return x


def _numpy_tuple_deep(x: np.ndarray) -> tuple:
    # return tuple(map(_numpy_tuple, x))
    return x


class MPMConfig(eqx.Module):
    inv_cell_size: float = eqx.field(static=True, converter=lambda x: float(x))

    origin: tuple = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))
    end: tuple = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))
    grid_size: tuple = eqx.field(static=True, converter=lambda x: _numpy_tuple(x))
    num_cells: int = eqx.field(static=True, converter=lambda x: int(x))
    cell_size: float = eqx.field(static=True, converter=lambda x: float(x))

    num_points: int = eqx.field(static=True, converter=lambda x: int(x))

    dt: float = eqx.field(static=True, converter=lambda x: float(x))
    num_steps: int = eqx.field(static=True, converter=lambda x: int(x))
    store_every: int = eqx.field(static=True, converter=lambda x: int(x))
    dim: int = eqx.field(static=True, converter=lambda x: int(x))

    shapefunction: str = eqx.field(static=True, converter=lambda x: str(x))
    forward_window: tuple = eqx.field(
        static=True, converter=lambda x: _numpy_tuple_deep(x)
    )
    backward_window: tuple = eqx.field(
        static=True, converter=lambda x: _numpy_tuple_deep(x)
    )
    window_size: int = eqx.field(static=True, converter=lambda x: int(x))
    padding: tuple = eqx.field(static=True)
    ppc: int = eqx.field(static=True, converter=lambda x: int(x))

    dir_path: str = eqx.field(static=True, converter=lambda x: str(x))

    project: str = eqx.field(static=True, converter=lambda x: str(x))

    def __init__(
        self: Self,
        origin: list,
        end: list,
        cell_size: float,
        num_points: int = 0,
        shapefunction: str = "linear",
        ppc=1,
        num_steps=0,
        store_every=0,
        dt=0.0,
        unroll_grid_kernels=True,
        default_gpu_id: int = None,
        project: str = "",
        **kwargs: Generic,
    ):
        self.inv_cell_size = 1.0 / cell_size
        self.grid_size = ((np.array(end) - np.array(origin)) / cell_size + 1).astype(
            int
        )
        self.cell_size = cell_size
        self.origin = np.array(origin)
        self.end = np.array(end)

        self.num_cells = np.prod(self.grid_size).astype(int)
        self.num_points = num_points
        self.dim = len(self.grid_size)

        self.ppc = ppc
        self.num_steps = num_steps
        self.store_every = store_every
        self.dt = dt

        if shapefunction == "linear":
            window_1D = np.arange(2).astype(int)
        elif shapefunction == "cubic":
            window_1D = np.arange(4).astype(int) - 1

        if self.dim == 2:
            self.forward_window = np.array(np.meshgrid(window_1D, window_1D)).T.reshape(
                -1, self.dim
            )
        elif self.dim == 3:
            self.forward_window = np.array(
                np.meshgrid(window_1D, window_1D, window_1D)
            ).T.reshape(-1, self.dim)
        elif self.dim == 1:
            self.forward_window = window_1D  # not tested!
            raise NotImplementedError

        self.backward_window = self.forward_window[::-1] - 1
        self.window_size = len(self.backward_window)
        self.shapefunction = shapefunction

        self.padding = (0, 3 - self.dim)

        if "file" in kwargs:
            file = kwargs.get("file")
        else:
            file = sys.argv[0]

        self.dir_path = os.path.dirname(file) + "/"

        self.project = project
        if default_gpu_id:
            set_default_gpu(default_gpu_id)

    def print_summary(self):
        print("=" * 50)
        print("Config summary")
        print("=" * 50)
        print(f"[MPMConfig] project = {self.project}")
        print(f"[MPMConfig] dim = {self.dim}")
        print(f"[MPMConfig] num_points = {self.num_points}")
        print(f"[MPMConfig] num_cells = {self.num_cells}")
        print(f"[MPMConfig] num_interactions = {self.num_points*self.window_size}")
        print(f"[MPMConfig] domain origin = {self.origin}")
        print(f"[MPMConfig] domain end = {self.end}")
        print("=" * 50)
