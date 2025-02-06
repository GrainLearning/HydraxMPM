import hydraxmpm as hdx
import jax.numpy as jnp
import jax


def test_create():
    """Test two ways of initializing solver"""

    solver = hdx.ETSolver(
        config=hdx.Config(total_time=1.0, dt=0.001),
        constitutive_law=hdx.LinearIsotropicElastic(E=1000, nu=0.2),
        et_benchmarks=hdx.VolumeControlShear(x_range=(0.0, 1.0), y_range=(0.0, 0.0)),
    )
    assert isinstance(solver, hdx.ETSolver)


def test_update():
    """Test to see if element test runs"""
    solver = hdx.ETSolver(
        config=hdx.Config(
            total_time=1.0,
            dt=0.001,
            output=("stress_stack", "p_stack", "phi_stack"),
        ),
        constitutive_law=hdx.LinearIsotropicElastic(
            E=1000, nu=0.2, phi_0=0.5, p_0=1000
        ),
        et_benchmarks=hdx.VolumeControlShear(x_range=(0.0, 1.0), y_range=(0.0, 0.0)),
    )
    solver = solver.setup()

    stress, p_stack, phi_stack = solver.run()
    # passes check
