"""Full-horizon forward-AD inverse analysis for granular collapse.

This module addresses the single-parameter problem: recover the friction angle
from one or more final global measures. The physical simulation always reaches
the requested terminal time. A single forward-mode JVP carries the friction-
angle tangent through the complete trajectory and returns the sensitivity of
every selected terminal measure. This script performs inverse optimization;
sensitivity and finite-difference diagnostics live in
``collapse_inverse_forward_ad_diagnostics.py``.
"""

import argparse
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Literal, cast

import jax
import jax.numpy as jnp
import optax

import hydraxmpm  # noqa: F401 - Avoid a lazy import during a JAX trace.
from projects.collapse.collapse import DT, TOTAL_TIME, simulate_collapse

jax.config.update("jax_enable_x64", True)

GlobalMeasureKey = Literal[
    "terminal_height",
    "terminal_com_x",
    "terminal_com_y",
    "terminal_runout",
]

GLOBAL_MEASURE_KEYS: tuple[GlobalMeasureKey, ...] = (
    "terminal_height",
    "terminal_com_x",
    "terminal_com_y",
    "terminal_runout",
)
DEFAULT_MEASURE_KEYS: tuple[GlobalMeasureKey, ...] = ("terminal_runout",)

REF_PHI = 20.0
REF_C0 = 10.0
PHI_BOUNDS = (5.0, 45.0)
TOTAL_STEPS = int(round(TOTAL_TIME / DT))
SCALE_FLOOR = 1.0e-12


def normalize_measure_keys(
    measure_keys: str | Sequence[str],
) -> tuple[GlobalMeasureKey, ...]:
    """Validate measure names and return a nonempty, duplicate-free tuple."""
    keys = (measure_keys,) if isinstance(measure_keys, str) else tuple(measure_keys)
    if not keys:
        raise ValueError("at least one global measure must be selected")

    invalid = tuple(key for key in keys if key not in GLOBAL_MEASURE_KEYS)
    if invalid:
        valid = ", ".join(GLOBAL_MEASURE_KEYS)
        raise ValueError(
            f"unsupported global measure(s) {invalid}; choose from: {valid}"
        )
    if len(set(keys)) != len(keys):
        raise ValueError("global measure keys must not contain duplicates")
    return cast(tuple[GlobalMeasureKey, ...], keys)


def _pack_global_measures(
    global_measures: dict[str, jax.Array],
    measure_keys: tuple[GlobalMeasureKey, ...],
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    """Pack selected scalar terminal measures in the requested order."""
    center_of_mass = global_measures["final_center_of_mass"]
    available = {
        "terminal_height": global_measures["final_height"],
        "terminal_com_x": center_of_mass[0],
        "terminal_com_y": center_of_mass[1],
        "terminal_runout": global_measures["final_runout_distance"],
    }
    return jnp.stack(tuple(available[key] for key in measure_keys)).astype(dtype)


def terminal_measures(
    fric_angle: jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    num_steps: int = TOTAL_STEPS,
) -> jax.Array:
    """Return selected final global measures after the requested trajectory."""
    keys = normalize_measure_keys(measure_keys)
    phi = jnp.asarray(fric_angle, dtype=jnp.float64)
    result = simulate_collapse(
        fric_angle=phi,
        c0=jnp.asarray(REF_C0, dtype=phi.dtype),
        num_steps=num_steps,
        compute_local=False,
    )
    return _pack_global_measures(result["global"], keys, dtype=phi.dtype)


@lru_cache(maxsize=None)
def compiled_terminal_measures(
    num_steps: int,
    measure_keys: tuple[GlobalMeasureKey, ...],
) -> Callable[[jax.Array], jax.Array]:
    """Build and cache a terminal-measure program for one static configuration."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return jax.jit(
        lambda phi: terminal_measures(
            phi,
            measure_keys=measure_keys,
            num_steps=num_steps,
        )
    )


def measures_and_jvp(
    fric_angle: float | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    num_steps: int = TOTAL_STEPS,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate selected terminal measures and all their friction sensitivities."""
    keys = normalize_measure_keys(measure_keys)
    phi = jnp.asarray(fric_angle, dtype=jnp.float64)
    measure_fn = compiled_terminal_measures(num_steps, keys)
    return jax.jvp(measure_fn, (phi,), (jnp.ones_like(phi),))


def loss_and_gradient(
    fric_angle: float | jax.Array,
    reference_measures: float | Sequence[float] | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    num_steps: int = TOTAL_STEPS,
    scales: float | Sequence[float] | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return normalized loss, gradient, measures, and measure sensitivities.

    For selected measures ``y_i``, references ``y_i*``, and scales ``s_i``,
    the equally weighted loss and its forward-mode gradient are

    ``L = 0.5 * mean(((y - y*) / s) ** 2)``

    ``dL/dphi = mean(((y - y*) / s) * (dy/dphi) / s)``.

    By default, each scale is the absolute value of its reference measure.
    A scalar ``scales`` value is broadcast to every selected measure.
    """
    keys = normalize_measure_keys(measure_keys)
    reference = jnp.atleast_1d(jnp.asarray(reference_measures, dtype=jnp.float64))
    expected_shape = (len(keys),)
    if reference.shape != expected_shape:
        raise ValueError(
            f"reference_measures must have shape {expected_shape}, "
            f"got {reference.shape}"
        )

    measures, sensitivities = measures_and_jvp(
        fric_angle,
        measure_keys=keys,
        num_steps=num_steps,
    )
    if scales is None:
        scale_values = jnp.maximum(jnp.abs(reference), SCALE_FLOOR)
    else:
        scale_values = jnp.atleast_1d(jnp.asarray(scales, dtype=reference.dtype))
        if scale_values.shape == (1,) and len(keys) > 1:
            scale_values = jnp.broadcast_to(scale_values, expected_shape)
        if scale_values.shape != expected_shape:
            raise ValueError(
                f"scales must have shape {expected_shape}, got {scale_values.shape}"
            )
        scale_values = jnp.maximum(jnp.abs(scale_values), SCALE_FLOOR)

    residuals = (measures - reference) / scale_values
    loss = 0.5 * jnp.mean(residuals**2)
    gradient = jnp.mean(residuals * sensitivities / scale_values)
    return loss, gradient, measures, sensitivities


def run_inverse_analysis(
    num_steps: int = 25,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    forward_steps: int = TOTAL_STEPS,
    phi_init: float = 10.0,
    learning_rate: float = 0.5,
) -> tuple[float, list[float]]:
    """Recover one friction angle from selected full-horizon terminal measures."""
    keys = normalize_measure_keys(measure_keys)
    measure_fn = compiled_terminal_measures(forward_steps, keys)
    reference = measure_fn(jnp.asarray(REF_PHI, dtype=jnp.float64))
    theta = jnp.asarray(phi_init, dtype=jnp.float64)
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta)
    history: list[float] = []

    print(
        f"reference phi={REF_PHI:.6f} deg | steps={forward_steps} | "
        f"measures={','.join(keys)}"
    )
    for key, value in zip(keys, reference, strict=True):
        print(f"  reference {key}={float(value):.12e}")

    for step in range(num_steps):
        loss, gradient, measures, sensitivities = loss_and_gradient(
            theta,
            reference,
            measure_keys=keys,
            num_steps=forward_steps,
        )
        updates, opt_state = optimizer.update(gradient, opt_state, theta)
        theta = jnp.clip(
            optax.apply_updates(theta, updates),
            PHI_BOUNDS[0],
            PHI_BOUNDS[1],
        )
        history.append(float(loss))
        print(
            f"step={step:03d} | loss={float(loss):.6e} | "
            f"phi={float(theta):.6f} deg | dL/dphi={float(gradient):.9e}"
        )
        for key, value, sensitivity in zip(keys, measures, sensitivities, strict=True):
            print(
                f"  {key}={float(value):.9e} | "
                f"d({key})/dphi={float(sensitivity):.9e}"
            )

    print(f"recovered friction angle: {float(theta):.9f} deg")
    return float(theta), history


def main() -> None:
    """Command-line entry point for single-parameter inverse analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-init", type=float, default=10.0)
    parser.add_argument("--forward-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument(
        "--measures",
        nargs="+",
        choices=GLOBAL_MEASURE_KEYS,
        default=list(DEFAULT_MEASURE_KEYS),
        help="one or more terminal global measures used by the inverse loss",
    )
    args = parser.parse_args()

    run_inverse_analysis(
        num_steps=args.iterations,
        measure_keys=args.measures,
        forward_steps=args.forward_steps,
        phi_init=args.phi_init,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
