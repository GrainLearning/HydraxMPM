def run_sim():
    import sys
    import os
    from functools import partial

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import jax
    import hydraxmpm as hdx
    import jax.numpy as jnp

    import equinox as eqx
    from collapse_setup import CollapseSetup, CollapseParams
    from floor import SpectralSetup
    from enum import Enum

    from save_data import save_compressed_sim

    import os
    from pathlib import Path

    base_path = Path(__file__).resolve().parent

    class Mode(Enum):
        VARY_ANGLE = 0

        VARY_FRIC = 1

        VARY_GEOM = 2


    mode = Mode.VARY_ANGLE
    N = 100
    num_harmonics = 16
    key = jax.random.PRNGKey(0)

    PARAM_SPECS = [
        {"name": "tilt_angle", "min": 0.0, "max": 30.0},
        {"name": "friction_base", "min": 0.0, "max": 2.0},
        # friction
        {"name": "friction_hurst", "min": 0.2, "max": 0.8},
        {"name": "friction_lambda_min", "min": 0.01, "max": 0.2},
        {"name": "friction_lambda_max", "min": 0.5, "max": 2.0},
        {"name": "friction_rms_height", "min": 0.01, "max": 0.1},
        # bumps
        {"name": "bump_hurst", "min": 0.2, "max": 0.8},
        {"name": "bump_lambda_min", "min": 0.01, "max": 0.2},
        {"name": "bump_lambda_max", "min": 0.5, "max": 2.0},
        {"name": "bump_rms_height", "min": 0.01, "max": 0.1},
    ]
    PARAM_NAMES = [p["name"] for p in PARAM_SPECS]

    LOWER_BOUNDS = jnp.array([p["min"] for p in PARAM_SPECS])
    UPPER_BOUNDS = jnp.array([p["max"] for p in PARAM_SPECS])
    num_params = len(PARAM_SPECS)

    population = jax.random.uniform(
        key, (N, num_params), minval=LOWER_BOUNDS, maxval=UPPER_BOUNDS
    )

    if mode == Mode.VARY_ANGLE:

        def create_collapse_params(param_vector, sim_id):
            param_dict = {name: value for name, value in zip(PARAM_NAMES, param_vector)}

            return CollapseParams(
                bumps=SpectralSetup(scale_base=0.0, rms_height=0.0, num_harmonics=1),
                frictions=SpectralSetup(
                    scale_base=0.5, rms_height=0.0, num_harmonics=1
                ),
                tilt_angle=param_dict["tilt_angle"],
                sim_id=sim_id,
            )

        output_dir = "output/collapse_vary_angle"

    elif mode == Mode.VARY_FRIC:

        def create_collapse_params(param_vector, sim_id):
            param_dict = {name: value for name, value in zip(PARAM_NAMES, param_vector)}

            frictions = SpectralSetup(
                scale_base=param_dict["friction_base"],
                lambda_min=param_dict["friction_lambda_min"],
                lambda_max=param_dict["friction_lambda_max"],
                rms_height=param_dict["friction_rms_height"],
                hurst=param_dict["friction_hurst"],
                num_harmonics=num_harmonics,
            )

            return CollapseParams(
                bumps=SpectralSetup(scale_base=0.0, rms_height=0.0, num_harmonics=1),
                frictions=frictions,
                tilt_angle=param_dict["tilt_angle"],
                sim_id=sim_id,
            )

        output_dir = "output/collapse_vary_fric"

    elif mode == Mode.VARY_GEOM:

        def create_collapse_params(param_vector, sim_id):
            param_dict = {name: value for name, value in zip(PARAM_NAMES, param_vector)}

            bumps = SpectralSetup(
                hurst=param_dict["bump_hurst"],
                lambda_min=param_dict["bump_lambda_min"],
                lambda_max=param_dict["bump_lambda_max"],
                rms_height=param_dict["bump_rms_height"],
                num_harmonics=num_harmonics,
            )

            frictions = SpectralSetup(
                scale_base=0.5, rms_height=0.0, num_harmonics=1
            )
            return CollapseParams(
                bumps=bumps,
                frictions=frictions,
                tilt_angle=param_dict["tilt_angle"],
                sim_id=sim_id,
            )

        output_dir = "output/collapse_vary_geom"

    sim_ids = jnp.arange(N)

    batch_configs = jax.vmap(create_collapse_params)(population, sim_ids)

    collapse_procedure = CollapseSetup()

    mpm_solver, sim_state = collapse_procedure.build_template()


    # uncomment these to enable viewer or VTK output during the simulation (will slow down the runs)
    # collapse_procedure.setup_viewer(
    #     mpm_solver, sim_state, batch_configs=batch_configs, N=N, grid_columns=2
    # )

    # collapse_procedure.setup_vtk(
    #     relative_dir=__file__,
    #     output_dir=output_dir,
    #     solver=mpm_solver,
    #     state=sim_state,
    #     batch_configs=batch_configs,
    #     N=N,
    # )

    final_output_path = base_path / output_dir

    final_output_path.mkdir(parents=True, exist_ok=True)

    batch_trajectories = jax.vmap(
        lambda p: collapse_procedure.run_with_trajectory(mpm_solver, sim_state, p)
    )(batch_configs)

    metadata_to_save = {
        "mode": mode.name,
        "N": int(N),
        "num_harmonics": int(num_harmonics),
        "key": int(key[0]),
        "parameter_map": {
            name: population[:, i] for i, name in enumerate(PARAM_NAMES)
        },
        
        "bounds": {
            "lower": LOWER_BOUNDS,
            "upper": UPPER_BOUNDS
        }
    }

    save_compressed_sim(
        batch_trajectories,
        path=final_output_path / "trajectories.zarr",
        metadata=metadata_to_save
    )

    print("Done with simulation...")


import multiprocessing

if __name__ == "__main__":

    p = multiprocessing.Process(target=run_sim, args=())
    p.start()
    try:
        p.join()
    except KeyboardInterrupt:

        # Force JAX to release GPU memory by terminating the process
        p.terminate()

        p.join()

        print("Worker dead. GPU memory released.")

    if p.exitcode == 0:
        print("Simulation finished naturally.")
    else:
        print(f"Process ended with code {p.exitcode}")
