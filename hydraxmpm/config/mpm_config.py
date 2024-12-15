import os
import sys
import time

import dataclasses

import equinox as eqx
import jax
import numpy as np
from typing_extensions import Generic, Self

from ..utils.jax_helpers import set_default_gpu


def _numpy_tuple(x: np.ndarray) -> tuple:
    assert x.ndim == 1
    return tuple([sub_x.item() for sub_x in x])


def _numpy_tuple_deep(x: np.ndarray) -> tuple:
    return tuple(map(_numpy_tuple, x))


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

    output_path: str = eqx.field(static=True, converter=lambda x: str(x))

    project: str = eqx.field(static=True, converter=lambda x: str(x))

    device: jax.Device = eqx.field(static=True)

    def __init__(
        self: Self,
        origin: list,
        end: list,
        cell_size: float,
        num_points: int = 0,
        shapefunction: str = "cubic",
        ppc: int = 1,
        num_steps: int = 0,
        store_every: int = 0,
        dt: float = 0.0,
        default_gpu_id: int = None,
        output_path: str = "output",
        project: str = "",
        device: int = None,
        **kwargs: Generic,
    ):
        """
        Args:
            origin: domain start
            end: domain end
            cell_size: cell size of grid
            num_points: number of material points. Defaults to 0.
            shapefunction: Shapefunction type,
                select:["cubic","linear"]. Defaults to "cubic".
            ppc: number of particles discretized per cell. Defaults to 1.
            num_steps: number of steps to run. Defaults to 0.
            store_every: output every nth step. Defaults to 0.
            dt: constant time step. Defaults to 0.0.
            default_gpu_id: default gpu to run on. Defaults to None.
            project: project name. Defaults to "".
            output_path: output path relative to dir_path. Defaults to "output".
            device: sharding (not implemented yet). Defaults to None.
        """

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
        
        self.output_path = output_path

        self.project = project

        self.device = device

        if default_gpu_id:
            set_default_gpu(default_gpu_id)

    def print_summary(self):
        """Print a basic summary of the config"""
        print("~" * 75)
        print("MPM config summary")
        print("~" * 75)
        print(f"project: {self.project}")
        print(f"dim: {self.dim}")
        print(f"num_points: {self.num_points}")
        print(f"num_cells: {self.num_cells}")
        print(f"num_interactions: {self.num_points*self.window_size}")
        print(f"domain origin: {self.origin}")
        print(f"domain end: {self.end}")
        print(f"dt: {self.dt}")
        print(f"total time: {self.dt*self.num_steps}")
        print("~" * 75)

        # TODO print sharding

    def replace(self,**kwargs: Generic):
        return dataclasses.replace(self,**kwargs)
    
    def back_up_output(self):
        output_path = f"{self.dir_path }/{self.output_path}"
        if os.path.exists(output_path):
            print(f'Moving existing output files into a backup directory\n')
            timestamp = os.path.getmtime(output_path)
            formatted_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(timestamp))
            path = output_path.rstrip('/')
            backup_dir = f'{path}_backup_{formatted_time}'
            os.makedirs(backup_dir, exist_ok=True)
            os.rename(output_path, backup_dir)