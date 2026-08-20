import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import optax

from projects.collapse.collapse import simulate_collapse


def measure_vector_from_global(global_measures):
    x_center = global_measures["final_center_of_mass"][0]
    y_center = global_measures["final_center_of_mass"][1]
    return jnp.asarray(
        [
            global_measures["final_height"],
            x_center,
            y_center,
            global_measures["final_runout_distance"],
        ],
        dtype=jnp.float32,
    )


def reference_measure_vector(fric_angle: float = 27.0, c0: float = 10.0):
    result = simulate_collapse(fric_angle=fric_angle, c0=c0)
    return measure_vector_from_global(result["global"])


def forward_measure_vector(params):
    fric_angle = params[0]
    c0 = params[1]
    result = simulate_collapse(fric_angle=fric_angle, c0=c0)
    return measure_vector_from_global(result["global"])


@jax.jit
def loss_fn_single(theta, ref_vec):
    params = jnp.array([theta[0], 10.0], dtype=jnp.float32)
    pred = forward_measure_vector(params)
    err = pred - ref_vec
    return jnp.mean(err ** 2) / (jnp.mean(ref_vec ** 2) + 1e-8)


@jax.jit
def loss_fn_two(params, ref_vec):
    pred = forward_measure_vector(params)
    err = pred - ref_vec
    return jnp.mean(err ** 2) / (jnp.mean(ref_vec ** 2) + 1e-8)


def run_inverse_example(num_steps: int = 25, mode: str = "single"):
    """Optimize against the actual collapse forward model defined in collapse.py."""
    ref_vec = reference_measure_vector(fric_angle=20.0, c0=10.0)

    if mode == "single":
        theta = jnp.array([10.0], dtype=jnp.float32)
        optimizer = optax.adam(learning_rate=1.0e-2)
        opt_state = optimizer.init(theta)
        history = []
        for step in range(num_steps):
            loss_value, grads = jax.value_and_grad(loss_fn_single)(theta, ref_vec)
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = jnp.clip(optax.apply_updates(theta, updates), 5.0, 40.0)
            history.append(float(loss_value))
            if step % 5 == 0 or step == num_steps - 1:
                print(f"step={step:02d} | loss={float(loss_value):.6e} | fric_angle={float(theta[0]):.3f} deg")
        print("\nRecovered single-parameter fit:", float(theta[0]), "deg")
        return float(theta[0]), history

    if mode == "two":
        params = jnp.array([10.0, 0.0], dtype=jnp.float32)
        optimizer = optax.adam(learning_rate=1.0e-2)
        opt_state = optimizer.init(params)
        history = []
        for step in range(num_steps):
            loss_value, grads = jax.value_and_grad(loss_fn_two)(params, ref_vec)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = jnp.clip(optax.apply_updates(params, updates), jnp.array([5.0, 0.0]), jnp.array([40.0, 2.0e4]))
            history.append(float(loss_value))
            if step % 5 == 0 or step == num_steps - 1:
                print(
                    f"step={step:02d} | loss={float(loss_value):.6e} | "
                    f"fric_angle={float(params[0]):.3f} deg | c0={float(params[1]):.3e} Pa"
                )
        print("\nRecovered two-parameter fit:", float(params[0]), "deg, c0=", float(params[1]), "Pa")
        return float(params[0]), float(params[1]), history

    raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    print("Running single-parameter inversion using the collapse forward model:")
    run_inverse_example(num_steps=1, mode="single")
    print("\nRunning two-parameter inversion using the collapse forward model:")
    run_inverse_example(num_steps=1, mode="two")
