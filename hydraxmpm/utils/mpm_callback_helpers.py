from typing import Dict, List, Optional, Tuple


import numpy as np
import pyvista as pv
import os

import warnings


def get_files(output_dr, prefix):
    all_files = [
        f for f in os.listdir(output_dr) if os.path.isfile(os.path.join(output_dr, f))
    ]

    selected_files = [x for x in all_files if prefix in x]

    selected_files = [x for x in selected_files if ".npz" in x]

    selected_files_sorted = sorted(selected_files, key=lambda x: int(x.split(".")[1]))

    return [output_dr + "/" + x for x in selected_files_sorted]


def give_3d(position_stack):
    _, dim = position_stack.shape
    return np.pad(
        position_stack,
        [(0, 0), (0, 3 - dim)],
        mode="constant",
        constant_values=0,
    )


def npz_to_vtk(
    input_folder,
    output_folder=None,
    remove_word_stack=False,
    verbose=False,
    kind="material_points",
):
    if type(kind) == str:
        kind = [kind]

    for k in kind:
        type_files = get_files(input_folder, k)

        for f in type_files:
            input_arrays = np.load(f)

            position_stack = input_arrays.get("position_stack", None)
            if position_stack is None:
                position_stack = input_arrays.get("grid_position_stack", None)

            if position_stack is None:
                warnings.warn(f"No position_stack found in {f}, skipping")
                continue

            position_stack = give_3d(position_stack)

            cloud = pv.PolyData(position_stack)
            print(f"Loaded {f} with {cloud.n_points} points")
            for arr in input_arrays.files:
                if arr == "grid_mesh":
                    continue
                if (remove_word_stack) and ("_stack" in arr):
                    arr = arr.split("_stack")[0]
                cloud[arr] = input_arrays[arr]

            head, tail = os.path.split(f)

            new_tail = ".".join(tail.split(".")[:2]) + ".vtk"
            outfile = os.path.join(output_folder, new_tail)
            cloud.save(outfile)
