"""AD-based inverse analysis for the collapse benchmark.

This script uses [simulate_collapse](/home/hcheng/GrainLearning/HydraxMPM/projects/collapse/collapse.py)
as the forward kernel and optimizes material parameters from global observables:
final height, center-of-mass x/y, and runout distance.
"""

import os
from typing import Literal

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import optax

from projects.collapse.collapse import simulate_collapse

Mode = Literal["single", "two"]

# Reference condition and AD-forward horizon.
REF_PHI = 20.0
REF_C0 = 10.0
FORWARD_STEPS = 100
EPS = 1e-8


def measure_vector(global_measures: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Pack global measures into a fixed 4-vector."""
    com = global_measures["final_center_of_mass"]
    return jnp.asarray(
        [
            global_measures["final_height"],
            com[0],
            com[1],
            global_measures["final_runout_distance"],
        ],
        dtype=jnp.float32,
    )


def normalised_mse(pred: jnp.ndarray, ref: jnp.ndarray) -> jnp.ndarray:
    """Scale-invariant loss on the global-measure vector."""
    err = pred - ref
    return jnp.mean(err**2) / (jnp.mean(ref**2) + EPS)


def run_forward(fric_angle: jnp.ndarray, c0: jnp.ndarray) -> jnp.ndarray:
    """Run forward collapse and return the global-measure vector."""
    result = simulate_collapse(fric_angle=fric_angle, c0=c0, num_steps=FORWARD_STEPS)
    return measure_vector(result["global"])


def loss_single(theta: jnp.ndarray, ref_vec: jnp.ndarray) -> jnp.ndarray:
    """Single-parameter loss with fixed cohesion."""
    pred = run_forward(theta[0], jnp.asarray(REF_C0, dtype=jnp.float32))
    return normalised_mse(pred, ref_vec)


def loss_two(params: jnp.ndarray, ref_vec: jnp.ndarray) -> jnp.ndarray:
    """Two-parameter loss in [friction_angle, cohesion]."""
    pred = run_forward(params[0], params[1])
    return normalised_mse(pred, ref_vec)


def build_reference_vector() -> jnp.ndarray:
    """Generate reference global measures from the forward model."""
    ref_result = simulate_collapse(fric_angle=REF_PHI, c0=REF_C0, num_steps=FORWARD_STEPS)
    return measure_vector(ref_result["global"])


def run_inverse_example(num_steps: int = 25, mode: Mode = "single"):
    """Run AD-based inverse optimization.

    Parameters
    ----------
    num_steps:
        Number of optimizer updates.
    mode:
        - ``single``: fit friction angle only.
        - ``two``: fit friction angle and cohesion jointly.
    """
    ref_vec = build_reference_vector()
    print(
        "Reference global measures "
        f"(phi={REF_PHI} deg, c0={REF_C0} Pa, forward_steps={FORWARD_STEPS}): {ref_vec}"
    )

    if mode == "single":
        theta = jnp.array([10.0], dtype=jnp.float32)
        bounds_lo = jnp.array([5.0], dtype=jnp.float32)
        bounds_hi = jnp.array([45.0], dtype=jnp.float32)
        optimizer = optax.adam(learning_rate=0.5)
        opt_state = optimizer.init(theta)
        history: list[float] = []

        for step in range(num_steps):
            loss_val, grads = jax.value_and_grad(loss_single)(theta, ref_vec)
            updates, opt_state = optimizer.update(grads, opt_state, theta)
            theta = jnp.clip(optax.apply_updates(theta, updates), bounds_lo, bounds_hi)
            history.append(float(loss_val))
            if step % 5 == 0 or step == num_steps - 1:
                print(
                    f"step={step:03d} | loss={float(loss_val):.6e} | "
                    f"fric_angle={float(theta[0]):.4f} deg"
                )

        print(f"\nRecovered single-parameter fit: {float(theta[0]):.6f} deg")
        return float(theta[0]), history

    if mode == "two":
        params = jnp.array([10.0, 200.0], dtype=jnp.float32)
        bounds_lo = jnp.array([5.0, 0.0], dtype=jnp.float32)
        bounds_hi = jnp.array([45.0, 500.0], dtype=jnp.float32)
        optimizer = optax.adam(learning_rate=jnp.array([0.5, 20.0], dtype=jnp.float32))
        opt_state = optimizer.init(params)
        history: list[float] = []

        for step in range(num_steps):
            loss_val, grads = jax.value_and_grad(loss_two)(params, ref_vec)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = jnp.clip(optax.apply_updates(params, updates), bounds_lo, bounds_hi)
            history.append(float(loss_val))
            if step % 5 == 0 or step == num_steps - 1:
                print(
                    f"step={step:03d} | loss={float(loss_val):.6e} | "
                    f"fric_angle={float(params[0]):.4f} deg | c0={float(params[1]):.4f} Pa"
                )

        print(
            "\nRecovered two-parameter fit: "
            f"fric_angle={float(params[0]):.6f} deg, c0={float(params[1]):.6f} Pa"
        )
        return float(params[0]), float(params[1]), history

    raise ValueError(f"Unsupported mode: {mode!r}. Use 'single' or 'two'.")


if __name__ == "__main__":
    print("=" * 72)
    print("Single-parameter inversion (friction angle)")
    print("=" * 72)
    run_inverse_example(num_steps=100, mode="single")

    print()
    print("=" * 72)
    print("Two-parameter inversion (friction angle + cohesion)")
    print("=" * 72)
    run_inverse_example(num_steps=100, mode="two")
