import os

from hydraxmpm.material_points import material_points
import itertools

from ..utils.mpm_callback_helpers import get_files


def view(output_dir, scalars=None, vminmaxs=None, refresh_rate=0.05):
    import polyscope as ps
    import numpy as np
    import time
    from pathlib import Path

    material_points_files = get_files(output_dir, "material_points")

    if len(material_points_files) == 0:
        print("No material_points files found")
        return

    ps.init()
    print("Loaded material_points files ", len(material_points_files))

    global mp_cycler
    mp_cycler = itertools.cycle(material_points_files)

    input_arrays = np.load(next(mp_cycler))

    position_stack = input_arrays.get("position_stack", None)

    ps.set_navigation_style("planar")

    ps.init()

    point_cloud = ps.register_point_cloud(
        "material_points", position_stack, enabled=True
    )

    if scalars is None:
        scalars = []
    if vminmaxs is None:
        vminmaxs = []

    for si, scalar in enumerate(scalars):
        data = input_arrays.get(scalar)
        if data is not None:
            point_cloud.add_scalar_quantity(
                scalar,
                data,
                # vminmax=vminmaxs[si]
            )

    forces_files = get_files(output_dir, "forces")
    global rp_cycler
    rp_cycler = itertools.cycle(forces_files)

    if len(forces_files) > 0:
        input_arrays = np.load(next(rp_cycler))
        r_position_stack = input_arrays.get("position_stack", None)
        r_point_cloud = ps.register_point_cloud(
            "rigid_points", r_position_stack, enabled=True
        )

        binary_mask = np.arange(r_position_stack.shape[0])

        r_point_cloud.add_scalar_quantity(
            "binary",
            binary_mask,
            enabled=True,
        )

    print("Polyscope viewer started")
    print("Press Ctrl+C to exit")

    def update():
        global mp_cycler, rp_cycler
        time.sleep(refresh_rate)
        mp_file = next(mp_cycler)

        mp_input_arrays = np.load(mp_file)

        position_stack = mp_input_arrays.get("position_stack", None)

        for scalar in scalars:
            data = mp_input_arrays.get(scalar)
            if data is not None:
                point_cloud.add_scalar_quantity(
                    scalar,
                    data,
                    # vminmax=vminmaxs[si]
                )
        point_cloud.update_point_positions(position_stack)
        if len(forces_files) > 0:
            forces_file = get_files(output_dir, "forces")

            if len(forces_file) > 0:
                input_arrays = np.load(next(rp_cycler))
                r_position_stack = input_arrays.get("position_stack", None)
                r_point_cloud.update_point_positions(r_position_stack)

    ps.set_user_callback(update)
    ps.show()
