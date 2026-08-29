"""Full-horizon forward-AD inverse analysis for granular collapse.

This module addresses the single-parameter problem: infer the friction angle
from one or more final global measures. The physical simulation always reaches
the requested terminal time. A single forward-mode JVP carries the friction-
angle tangent through the complete trajectory and returns the sensitivity of
every selected terminal measure.
"""

import argparse
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np
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
ITERATION_DATA_DIR = Path(__file__).resolve().parent / "output" / "inverse_iterations"
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_EARLY_STOPPING_PATIENCE = 50
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1.0e-10


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


def normalize_measure_weights(
    measure_weights: float | Sequence[float] | jax.Array | None,
    *,
    num_measures: int,
    dtype: jnp.dtype = jnp.float64,
) -> jax.Array:
    """Validate and return one nonnegative loss weight per selected measure."""
    if num_measures <= 0:
        raise ValueError("num_measures must be positive")
    if measure_weights is None:
        return jnp.ones((num_measures,), dtype=dtype)

    weights = jnp.atleast_1d(jnp.asarray(measure_weights, dtype=dtype))
    expected_shape = (num_measures,)
    if weights.shape == (1,) and num_measures > 1:
        weights = jnp.broadcast_to(weights, expected_shape)
    if weights.shape != expected_shape:
        raise ValueError(
            f"measure_weights must have shape {expected_shape}, got {weights.shape}"
        )
    if bool(jnp.any(~jnp.isfinite(weights))):
        raise ValueError("measure_weights must be finite")
    if bool(jnp.any(weights < 0.0)):
        raise ValueError("measure_weights must be nonnegative")
    if float(jnp.sum(weights)) <= 0.0:
        raise ValueError("at least one measure weight must be positive")
    return weights


def loss_and_gradient(
    fric_angle: float | jax.Array,
    reference_measures: float | Sequence[float] | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    num_steps: int = TOTAL_STEPS,
    scales: float | Sequence[float] | jax.Array | None = None,
    measure_weights: float | Sequence[float] | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return normalized loss, gradient, measures, and measure sensitivities.

    For selected measures ``y_i``, references ``y_i*``, and scales ``s_i``,
    weights ``w_i``, the normalized loss and its forward-mode gradient are

    ``L = 0.5 * sum(w * ((y - y*) / s) ** 2) / sum(w)``

    ``dL/dphi = sum(w * ((y-y*)/s) * (dy/dphi)/s) / sum(w)``.

    By default, each scale is the absolute value of its reference measure.
    Scalar scales and weights are broadcast to every selected measure. Equal
    positive weights reproduce the previous equally weighted mean.
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
    weight_values = normalize_measure_weights(
        measure_weights,
        num_measures=len(keys),
        dtype=reference.dtype,
    )
    weight_sum = jnp.sum(weight_values)
    loss = 0.5 * jnp.sum(weight_values * residuals**2) / weight_sum
    gradient = (
        jnp.sum(weight_values * residuals * sensitivities / scale_values)
        / weight_sum
    )
    return loss, gradient, measures, sensitivities


def save_iteration_state(
    iteration: int,
    *,
    loss: float | jax.Array,
    friction_angle: float | jax.Array,
    forward_steps: int,
    output_dir: str | Path = ITERATION_DATA_DIR,
) -> Path:
    """Save one consistent loss, parameter, and terminal field checkpoint.

    The saved loss and friction angle are the values at which the optimizer
    gradient was evaluated. Computing the volume-fraction field requires one
    additional primal collapse simulation; it is not included in the AD loss.
    """
    from projects.collapse.collapse_inverse_forward_ad_plots import (
        simulate_volume_fraction,
    )

    phi = float(friction_angle)
    x, y, volume_fraction, crop_extent = simulate_volume_fraction(
        phi,
        cohesion=REF_C0,
        num_steps=forward_steps,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"collapse_inverse_iteration_{iteration:04d}.npz"
    np.savez_compressed(
        path,
        iteration=np.asarray(iteration, dtype=np.int64),
        loss=np.asarray(float(loss), dtype=np.float64),
        friction_angle=np.asarray(phi, dtype=np.float64),
        x=x,
        y=y,
        volume_fraction=volume_fraction,
        crop_extent=np.asarray(crop_extent, dtype=np.float64),
    )
    return path


def run_inverse_analysis(
    num_steps: int = 25,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    measure_weights: float | Sequence[float] | jax.Array | None = None,
    forward_steps: int = TOTAL_STEPS,
    phi_init: float = 10.0,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    save_plots: bool = True,
    figure_dir: str | Path | None = None,
    save_iterations: bool = True,
    iteration_dir: str | Path = ITERATION_DATA_DIR,
) -> tuple[float, list[float]]:
    """Recover one friction angle and save optional plots and checkpoints."""
    keys = normalize_measure_keys(measure_keys)
    if early_stopping_patience == 0:
        early_stopping_patience = None
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    measure_fn = compiled_terminal_measures(forward_steps, keys)
    reference = measure_fn(jnp.asarray(REF_PHI, dtype=jnp.float64))
    weight_values = normalize_measure_weights(
        measure_weights,
        num_measures=len(keys),
        dtype=reference.dtype,
    )
    theta = jnp.clip(
        jnp.asarray(phi_init, dtype=jnp.float64),
        PHI_BOUNDS[0],
        PHI_BOUNDS[1],
    )
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(theta)
    history: list[float] = []
    friction_angle_history: list[float] = []
    best_loss = float("inf")
    best_theta = float(theta)
    iterations_without_improvement = 0
    stopped_early = False

    print(
        f"reference phi={REF_PHI:.6f} deg | steps={forward_steps} | "
        f"measures={','.join(keys)}"
    )
    for key, value, weight in zip(keys, reference, weight_values, strict=True):
        print(
            f"  reference {key}={float(value):.12e} | "
            f"loss_weight={float(weight):.6g}"
        )
    print(f"learning rate={learning_rate:.6e}")
    if early_stopping_patience is not None:
        print(
            f"early stopping patience={early_stopping_patience} | "
            f"minimum improvement={early_stopping_min_delta:.3e}"
        )

    for step in range(num_steps):
        evaluated_theta = float(theta)
        loss, gradient, measures, sensitivities = loss_and_gradient(
            theta,
            reference,
            measure_keys=keys,
            num_steps=forward_steps,
            measure_weights=weight_values,
        )
        if save_iterations:
            checkpoint_path = save_iteration_state(
                step,
                loss=loss,
                friction_angle=theta,
                forward_steps=forward_steps,
                output_dir=iteration_dir,
            )
            print(f"  saved iteration state: {checkpoint_path}")
        loss_value = float(loss)
        history.append(loss_value)
        friction_angle_history.append(evaluated_theta)
        print(
            f"step={step:03d} | loss={loss_value:.6e} | "
            f"phi={evaluated_theta:.6f} deg | dL/dphi={float(gradient):.9e} | "
            f"lr={learning_rate:.6e}"
        )
        for key, value, sensitivity in zip(keys, measures, sensitivities, strict=True):
            print(
                f"  {key}={float(value):.9e} | "
                f"d({key})/dphi={float(sensitivity):.9e}"
            )

        if loss_value < best_loss - early_stopping_min_delta:
            best_loss = loss_value
            best_theta = evaluated_theta
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        if (
            early_stopping_patience is not None
            and iterations_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"early stopping at step {step}: no loss improvement greater "
                f"than {early_stopping_min_delta:.3e} for "
                f"{early_stopping_patience} evaluations"
            )
            break

        updates, opt_state = optimizer.update(gradient, opt_state, theta)
        theta = jnp.clip(
            optax.apply_updates(theta, updates),
            PHI_BOUNDS[0],
            PHI_BOUNDS[1],
        )

    theta = jnp.asarray(best_theta, dtype=jnp.float64)
    stop_reason = "early stopping" if stopped_early else "iteration limit"
    print(
        f"best loss={best_loss:.12e} | recovered friction angle="
        f"{best_theta:.9f} deg | stop={stop_reason}"
    )
    if save_plots:
        from projects.collapse.collapse_inverse_forward_ad_plots import (
            FIGURES_DIR,
            create_inverse_plots,
        )

        create_inverse_plots(
            loss_history=history,
            friction_angle_history=friction_angle_history,
            identified_friction_angle=float(theta),
            reference_friction_angle=REF_PHI,
            cohesion=REF_C0,
            num_steps=forward_steps,
            output_dir=FIGURES_DIR if figure_dir is None else figure_dir,
        )
    return float(theta), history


def main() -> None:
    """Command-line entry point for single-parameter inverse analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-init", type=float, default=10.0)
    parser.add_argument("--forward-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=DEFAULT_EARLY_STOPPING_PATIENCE,
        help="evaluations without improvement before stopping; 0 disables",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=DEFAULT_EARLY_STOPPING_MIN_DELTA,
        help="minimum absolute loss decrease counted as an improvement",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="directory for inverse figures (default: collapse/output/figures)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="skip final summary figures and their two field simulations",
    )
    parser.add_argument(
        "--iteration-dir",
        type=Path,
        default=ITERATION_DATA_DIR,
        help="directory for per-iteration loss, parameter, and field checkpoints",
    )
    parser.add_argument(
        "--no-iteration-data",
        action="store_true",
        help="skip per-iteration volume-fraction checkpoints",
    )
    parser.add_argument(
        "--measures",
        nargs="+",
        choices=GLOBAL_MEASURE_KEYS,
        default=list(DEFAULT_MEASURE_KEYS),
        help="one or more terminal global measures used by the inverse loss",
    )
    parser.add_argument(
        "--measure-weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "nonnegative loss weights aligned with --measures; "
            "default: equal weights"
        ),
    )
    args = parser.parse_args()

    run_inverse_analysis(
        num_steps=args.iterations,
        measure_keys=args.measures,
        measure_weights=args.measure_weights,
        forward_steps=args.forward_steps,
        phi_init=args.phi_init,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_plots=not args.no_plots,
        figure_dir=args.figure_dir,
        save_iterations=not args.no_iteration_data,
        iteration_dir=args.iteration_dir,
    )


if __name__ == "__main__":
    main()
