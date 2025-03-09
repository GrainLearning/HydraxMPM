from typing import Dict, List, Optional, Tuple


import numpy as np
import pyvista as pv
import os

import warnings


def give_3d(position_stack):
    _, dim = position_stack.shape
    return np.pad(
        position_stack,
        [(0, 0), (0, 3 - dim)],
        mode="constant",
        constant_values=0,
    )


def npz_to_vtk(
    input_folder, output_folder=None, remove_word_stack=False, verbose=False
):
    files = [
        f
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f))
    ]
    files = [f for f in files if ".npz" in f]
    if output_folder is None:
        output_folder = input_folder

    for f in files:
        input_arrays = np.load(input_folder + "/" + f)

        position_stack = input_arrays.get("position_stack", None)
        if position_stack is None:
            position_stack = input_arrays.get("p2g_position_stack", None)

        if position_stack is None:
            warnings.warn(f"No position_stack found in {f}, skipping")
            continue

        position_stack = give_3d(position_stack)

        cloud = pv.PolyData(position_stack)
        for arr in input_arrays.files:
            if (remove_word_stack) and ("_stack" in arr):
                arr = arr.split("_stack")[0]
            cloud[arr] = input_arrays[arr]

        new_f = ".".join(f.split(".")[:2]) + ".vtk"

        cloud.save(output_folder + new_f)
