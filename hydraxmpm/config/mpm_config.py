from typing_extensions import Self

import equinox as eqx
import jax
import numpy as np
import os

from ..shapefunctions.shapefunctions import SHAPEFUNCTION

from ..utils.jax_helpers import set_default_gpu

import sys


class MPMConfig(eqx.Module):
    inv_cell_size: float = eqx.field(static=True, converter=lambda x: float(x))
    origin: tuple = eqx.field(static=True, converter=lambda x: tuple(x))
    end: tuple = eqx.field(static=True, converter=lambda x: tuple(x))
    grid_size: tuple = eqx.field(static=True, converter=lambda x: tuple(x))
    num_cells: int = eqx.field(static=True, converter=lambda x: int(x))
    cell_size: float = eqx.field(static=True, converter=lambda x: float(x))

    num_points: int = eqx.field(static=True, converter=lambda x: int(x))

    dt: float = eqx.field(static=True, converter=lambda x: float(x))
    num_steps: int = eqx.field(static=True, converter=lambda x: int(x))
    store_every: int = eqx.field(static=True, converter=lambda x: int(x))
    dim: int = eqx.field(static=True, converter=lambda x: int(x))

    shapefunction: SHAPEFUNCTION = eqx.field(static=True)
    forward_window: tuple = eqx.field(
        static=True, converter=lambda x: tuple(map(tuple, x))
    )
    backward_window: tuple = eqx.field(
        static=True, converter=lambda x: tuple(map(tuple, x))
    )
    window_size: int = eqx.field(static=True, converter=lambda x: int(x))
    padding: tuple = eqx.field(static=True)
    ppc: int = eqx.field(static=True, converter=lambda x: int(x))

    unroll_grid_kernels: bool = eqx.field(static=True, converter=lambda x: bool(x))

    dir_path: str = eqx.field(static=True, converter=lambda x: str(x))

    def __init__(
        self: Self,
        origin: list,
        end: list,
        cell_size: float,
        num_points: int,
        shapefunction: SHAPEFUNCTION = SHAPEFUNCTION.linear,
        ppc=1,
        num_steps=0,
        store_every=0,
        dt=0.0,
        unroll_grid_kernels=True,
        default_gpu_id: int = None,
    ):
        jax.debug.print(
            "Ignore the UserWarning from, the behavior is intended and expected."
        )

        self.inv_cell_size = 1.0 / cell_size
        self.grid_size = ((np.array(end) - np.array(origin)) / cell_size + 1).astype(
            int
        )
        self.cell_size = cell_size
        self.origin = origin
        self.end = end

        self.num_cells = np.prod(self.grid_size).astype(int)
        self.num_points = num_points
        self.dim = len(self.grid_size)

        self.ppc = ppc
        self.num_steps = num_steps
        self.store_every = store_every
        self.dt = dt

        if shapefunction == SHAPEFUNCTION.linear:
            window_1D = np.arange(2).astype(int)
        elif shapefunction == SHAPEFUNCTION.cubic:
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

        self.unroll_grid_kernels = unroll_grid_kernels

        self.padding = (0, 3 - self.dim)

        file = sys.argv[0]
        self.dir_path = os.path.dirname(file) + "/"

        if default_gpu_id:
            set_default_gpu(default_gpu_id)

    def print_summary(self):
        
        print(f"[MPMConfig] num_points = {self.num_points}")
        print(f"[MPMConfig] num_cells = {self.num_cells}") 
        print(f"[MPMConfig] num_interactions = {self.num_points*self.window_size}")