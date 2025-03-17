def view(config, scalars=None, vminmaxs=None, refresh_rate=0.05):
    import polyscope as ps
    import numpy as np
    import time
    from pathlib import Path

    ps.init()

    while True:
        time.sleep(2)
        try:
            print("trying to load..")
            input_arrays = np.load(config.output_path + "/material_points.0.npz")
            break
        except Exception:
            pass
    print("Loaded")
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

    step = 0

    filepath = Path(config.output_path + "/forces.0.npz")
    load_rigid = False
    if filepath.exists():
        input_arrays = np.load(config.output_path + "/forces.0.npz")

        r_position_stack = input_arrays.get("position_stack", None)
        r_point_cloud = ps.register_point_cloud(
            "rigid_points", r_position_stack, enabled=True
        )
        load_rigid = True

    def update():
        global step
        time.sleep(refresh_rate)
        try:
            cfile = config.output_path + f"/material_points.{step}.npz"
            print("loading", cfile)
            input_arrays = np.load(cfile)
            position_stack = input_arrays.get("position_stack", None)

            for scalar in scalars:
                data = input_arrays.get(scalar)
                if data is not None:
                    point_cloud.add_scalar_quantity(
                        scalar,
                        data,
                        # vminmax=vminmaxs[si]
                    )
            point_cloud.update_point_positions(position_stack)

            if load_rigid:
                input_arrays = np.load(config.output_path + f"/forces.{step}.npz")
                r_point_cloud.update_point_positions(input_arrays.get("position_stack"))

            if step >= config.num_steps:
                step = 0
            else:
                step += config.store_every
        except Exception as e:
            step = 0

    ps.set_user_callback(update)
    ps.show()
