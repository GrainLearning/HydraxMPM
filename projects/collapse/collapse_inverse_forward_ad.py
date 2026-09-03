"""Full-horizon forward-AD inverse analysis for granular collapse.

This module infers one or more material parameters from final global measures.
The physical simulation always reaches the requested terminal time. Forward-
mode JVPs carry each selected parameter tangent through the complete trajectory
and return the Jacobian of every selected terminal measure.
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
from projects.collapse.collapse import (
    BULK_DENSITY,
    BULK_HEIGHT_CUTOFF,
    CELL_SIZE,
    COLUMN_HEIGHT,
    DT,
    END,
    INITIAL_SOLID_VOLUME_FRACTION,
    LATERAL_STRESS_RATIO,
    ORIGIN,
    TOTAL_TIME,
    simulate_collapse,
)
from projects.collapse.utilities import (
    build_restart_payload,
    load_restart_state,
    save_iteration_state,
)

jax.config.update("jax_enable_x64", True)

GlobalMeasureKey = Literal[
    "terminal_height",
    "terminal_com_x",
    "terminal_com_y",
    "terminal_runout",
    "terminal_height_profile",
]
ParameterKey = Literal["friction_angle", "cohesion"]

GLOBAL_MEASURE_KEYS: tuple[GlobalMeasureKey, ...] = (
    "terminal_height",
    "terminal_com_x",
    "terminal_com_y",
    "terminal_runout",
    "terminal_height_profile",
)
DEFAULT_MEASURE_KEYS: tuple[GlobalMeasureKey, ...] = (
    "terminal_height_profile",
)
PARAMETER_KEYS: tuple[ParameterKey, ...] = ("friction_angle", "cohesion")
DEFAULT_PARAMETER_KEYS: tuple[ParameterKey, ...] = ("friction_angle",)

REF_PHI = 20.0
REF_C0 = 150.0
PHI_BOUNDS = (5.0, 45.0)
COHESION_BOUNDS = (0.0, 500.0)
TOTAL_STEPS = int(round(TOTAL_TIME / DT))
SCALE_FLOOR = 1.0e-12
ITERATION_DATA_DIR = Path(__file__).resolve().parent / "output" / "inverse_iterations"
DEFAULT_LEARNING_RATE = 0.5
DEFAULT_COHESION_LEARNING_RATE = 5.0
DEFAULT_EARLY_STOPPING_PATIENCE = 50
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1.0e-10
DEFAULT_CHECKPOINT_FREQUENCY = 25
HEIGHT_PROFILE_SIZE = int(round((END[0] - ORIGIN[0]) / CELL_SIZE)) + 1


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


def normalize_parameter_keys(
    parameter_keys: str | Sequence[str],
) -> tuple[ParameterKey, ...]:
    """Validate parameter names and return a nonempty, duplicate-free tuple."""
    keys = (
        (parameter_keys,)
        if isinstance(parameter_keys, str)
        else tuple(parameter_keys)
    )
    if not keys:
        raise ValueError("at least one material parameter must be selected")
    invalid = tuple(key for key in keys if key not in PARAMETER_KEYS)
    if invalid:
        valid = ", ".join(PARAMETER_KEYS)
        raise ValueError(f"unsupported parameter(s) {invalid}; choose from: {valid}")
    if len(set(keys)) != len(keys):
        raise ValueError("parameter keys must not contain duplicates")
    return cast(tuple[ParameterKey, ...], keys)


def _parameter_vector(
    parameters: float | Sequence[float] | jax.Array,
    parameter_keys: tuple[ParameterKey, ...],
) -> jax.Array:
    """Convert selected material parameters to a consistently shaped vector."""
    values = jnp.atleast_1d(jnp.asarray(parameters, dtype=jnp.float64))
    expected_shape = (len(parameter_keys),)
    if values.shape != expected_shape:
        raise ValueError(
            f"parameters must have shape {expected_shape}, got {values.shape}"
        )
    return values


def _unpack_parameters(
    parameters: jax.Array,
    parameter_keys: tuple[ParameterKey, ...],
) -> tuple[jax.Array, jax.Array]:
    """Combine selected values with fixed reference values."""
    values = {key: parameters[index] for index, key in enumerate(parameter_keys)}
    dtype = parameters.dtype
    return (
        values.get("friction_angle", jnp.asarray(REF_PHI, dtype=dtype)),
        values.get("cohesion", jnp.asarray(REF_C0, dtype=dtype)),
    )


def measure_component_sizes(
    measure_keys: tuple[GlobalMeasureKey, ...],
) -> tuple[int, ...]:
    """Return the flattened component count for every selected observable."""
    return tuple(
        HEIGHT_PROFILE_SIZE if key == "terminal_height_profile" else 1
        for key in measure_keys
    )


def measure_component_slices(
    measure_keys: tuple[GlobalMeasureKey, ...],
) -> tuple[slice, ...]:
    """Return slices into the flattened terminal-observable vector."""
    sizes = measure_component_sizes(measure_keys)
    starts = np.cumsum((0, *sizes[:-1]))
    return tuple(
        slice(int(start), int(start + size))
        for start, size in zip(starts, sizes, strict=True)
    )


def _pack_terminal_measures(
    global_measures: dict[str, jax.Array],
    height_profile: jax.Array | None,
    measure_keys: tuple[GlobalMeasureKey, ...],
    *,
    dtype: jnp.dtype,
) -> jax.Array:
    """Flatten selected scalar and profile observables in requested order."""
    center_of_mass = global_measures["final_center_of_mass"]
    available = {
        "terminal_height": global_measures["final_height"],
        "terminal_com_x": center_of_mass[0],
        "terminal_com_y": center_of_mass[1],
        "terminal_runout": global_measures["final_runout_distance"],
    }
    parts = []
    for key in measure_keys:
        if key == "terminal_height_profile":
            if height_profile is None:
                raise ValueError("height profile was not computed")
            parts.append(jnp.asarray(height_profile, dtype=dtype))
        else:
            parts.append(jnp.atleast_1d(jnp.asarray(available[key], dtype=dtype)))
    return jnp.concatenate(parts)


def terminal_measures(
    parameters: float | Sequence[float] | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    parameter_keys: str | Sequence[str] = DEFAULT_PARAMETER_KEYS,
    num_steps: int = TOTAL_STEPS,
) -> jax.Array:
    """Return selected final global measures after the requested trajectory."""
    keys = normalize_measure_keys(measure_keys)
    selected_parameters = normalize_parameter_keys(parameter_keys)
    theta = _parameter_vector(parameters, selected_parameters)
    phi, cohesion = _unpack_parameters(theta, selected_parameters)
    result = simulate_collapse(
        fric_angle=phi,
        c0=cohesion,
        num_steps=num_steps,
        compute_local=False,
        compute_height_profile="terminal_height_profile" in keys,
    )
    return _pack_terminal_measures(
        result["global"],
        result["height_profile"],
        keys,
        dtype=phi.dtype,
    )


@lru_cache(maxsize=None)
def compiled_terminal_measures(
    num_steps: int,
    measure_keys: tuple[GlobalMeasureKey, ...],
    parameter_keys: tuple[ParameterKey, ...] = DEFAULT_PARAMETER_KEYS,
) -> Callable[[jax.Array], jax.Array]:
    """Build and cache a terminal-measure program for one static configuration."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return jax.jit(
        lambda theta: terminal_measures(
            theta,
            measure_keys=measure_keys,
            parameter_keys=parameter_keys,
            num_steps=num_steps,
        )
    )


def measures_and_jacobian(
    parameters: float | Sequence[float] | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    parameter_keys: str | Sequence[str] = DEFAULT_PARAMETER_KEYS,
    num_steps: int = TOTAL_STEPS,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate terminal measures and their forward-mode parameter Jacobian."""
    measures_selected = normalize_measure_keys(measure_keys)
    parameters_selected = normalize_parameter_keys(parameter_keys)
    theta = _parameter_vector(parameters, parameters_selected)
    measure_fn = compiled_terminal_measures(
        num_steps,
        measures_selected,
        parameters_selected,
    )
    measures = measure_fn(theta)
    columns = tuple(
        jax.jvp(measure_fn, (theta,), (basis,))[1]
        for basis in jnp.eye(len(parameters_selected), dtype=theta.dtype)
    )
    return measures, jnp.stack(columns, axis=1)


def measures_and_jvp(
    fric_angle: float | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    num_steps: int = TOTAL_STEPS,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate selected terminal measures and all their friction sensitivities."""
    measures, jacobian = measures_and_jacobian(
        fric_angle,
        measure_keys=measure_keys,
        parameter_keys=DEFAULT_PARAMETER_KEYS,
        num_steps=num_steps,
    )
    return measures, jacobian[:, 0]


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
    parameters: float | Sequence[float] | jax.Array,
    reference_measures: float | Sequence[float] | jax.Array,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    parameter_keys: str | Sequence[str] = DEFAULT_PARAMETER_KEYS,
    num_steps: int = TOTAL_STEPS,
    scales: float | Sequence[float] | jax.Array | None = None,
    measure_weights: float | Sequence[float] | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return normalized loss, gradient, measures, and measure sensitivities.

    For selected measures ``y_i``, references ``y_i*``, and scales ``s_i``,
    weights ``w_i``, the normalized loss and its forward-mode gradient are

    ``L = 0.5 * sum(w * ((y - y*) / s) ** 2) / sum(w)``

    ``dL/dphi = sum(w * ((y-y*)/s) * (dy/dphi)/s) / sum(w)``.

    Scalar measures use their reference magnitude as the default scale. Height-
    profile components use the initial column height. A measure's loss weight
    is divided equally among all of its flattened components, so a profile does
    not gain weight merely because it contains many horizontal samples.
    """
    keys = normalize_measure_keys(measure_keys)
    reference = jnp.atleast_1d(jnp.asarray(reference_measures, dtype=jnp.float64))
    component_sizes = measure_component_sizes(keys)
    component_slices = measure_component_slices(keys)
    expected_shape = (sum(component_sizes),)
    if reference.shape != expected_shape:
        raise ValueError(
            f"reference_measures must have shape {expected_shape}, "
            f"got {reference.shape}"
        )

    selected_parameters = normalize_parameter_keys(parameter_keys)
    measures, jacobian = measures_and_jacobian(
        parameters,
        measure_keys=keys,
        parameter_keys=selected_parameters,
        num_steps=num_steps,
    )
    if scales is None:
        default_scale_parts = []
        for key, component_slice in zip(keys, component_slices, strict=True):
            if key == "terminal_height_profile":
                default_scale_parts.append(
                    jnp.full(
                        (component_slice.stop - component_slice.start,),
                        COLUMN_HEIGHT,
                        dtype=reference.dtype,
                    )
                )
            else:
                default_scale_parts.append(jnp.abs(reference[component_slice]))
        scale_values = jnp.maximum(
            jnp.concatenate(default_scale_parts), SCALE_FLOOR
        )
    else:
        scale_values = jnp.atleast_1d(jnp.asarray(scales, dtype=reference.dtype))
        if scale_values.shape == (1,) and expected_shape != (1,):
            scale_values = jnp.broadcast_to(scale_values, expected_shape)
        if scale_values.shape != expected_shape:
            raise ValueError(
                f"scales must have shape {expected_shape}, got {scale_values.shape}"
            )
        scale_values = jnp.maximum(jnp.abs(scale_values), SCALE_FLOOR)

    residuals = (measures - reference) / scale_values
    measure_weight_values = normalize_measure_weights(
        measure_weights,
        num_measures=len(keys),
        dtype=reference.dtype,
    )
    positive_components = sum(
        size
        for size, weight in zip(
            component_sizes, measure_weight_values, strict=True
        )
        if float(weight) > 0.0
    )
    if positive_components < len(selected_parameters):
        raise ValueError(
            "the number of positively weighted measures must be at least the "
            "number of inferred parameters"
        )
    component_weights = jnp.concatenate(
        tuple(
            jnp.full(
                (size,),
                measure_weight_values[index] / size,
                dtype=reference.dtype,
            )
            for index, size in enumerate(component_sizes)
        )
    )
    weight_sum = jnp.sum(component_weights)
    loss = 0.5 * jnp.sum(component_weights * residuals**2) / weight_sum
    gradient = jnp.sum(
        component_weights[:, None]
        * residuals[:, None]
        * jacobian
        / scale_values[:, None],
        axis=0,
    ) / weight_sum
    if len(selected_parameters) == 1:
        return loss, gradient[0], measures, jacobian[:, 0]
    return loss, gradient, measures, jacobian


def run_inverse_analysis(
    num_steps: int = 25,
    *,
    measure_keys: str | Sequence[str] = DEFAULT_MEASURE_KEYS,
    measure_weights: float | Sequence[float] | jax.Array | None = None,
    parameter_keys: str | Sequence[str] = DEFAULT_PARAMETER_KEYS,
    forward_steps: int = TOTAL_STEPS,
    phi_init: float = 10.0,
    cohesion_init: float = 100.0,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    cohesion_learning_rate: float = DEFAULT_COHESION_LEARNING_RATE,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    save_plots: bool = True,
    figure_dir: str | Path | None = None,
    save_iterations: bool = True,
    checkpoint_frequency: int = DEFAULT_CHECKPOINT_FREQUENCY,
    iteration_dir: str | Path = ITERATION_DATA_DIR,
    resume_from: str | Path | None = None,
) -> tuple[float | dict[ParameterKey, float], list[float]]:
    """Recover selected material parameters and save plots and checkpoints."""
    keys = normalize_measure_keys(measure_keys)
    selected_parameters = normalize_parameter_keys(parameter_keys)
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if checkpoint_frequency <= 0:
        raise ValueError("checkpoint_frequency must be positive")
    if early_stopping_patience == 0:
        early_stopping_patience = None
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be nonnegative")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if cohesion_learning_rate <= 0.0:
        raise ValueError("cohesion_learning_rate must be positive")

    reference_values = {"friction_angle": REF_PHI, "cohesion": REF_C0}
    initial_values = {"friction_angle": phi_init, "cohesion": cohesion_init}
    bounds = {"friction_angle": PHI_BOUNDS, "cohesion": COHESION_BOUNDS}
    rates = {
        "friction_angle": learning_rate,
        "cohesion": cohesion_learning_rate,
    }
    reference_theta = jnp.asarray(
        [reference_values[key] for key in selected_parameters], dtype=jnp.float64
    )
    weight_values = normalize_measure_weights(
        measure_weights,
        num_measures=len(keys),
        dtype=jnp.float64,
    )
    positive_components = sum(
        size
        for size, weight in zip(
            measure_component_sizes(keys), weight_values, strict=True
        )
        if float(weight) > 0.0
    )
    if positive_components < len(selected_parameters):
        raise ValueError(
            "the number of positively weighted measures must be at least the "
            "number of inferred parameters"
        )
    measure_fn = compiled_terminal_measures(
        forward_steps, keys, selected_parameters
    )
    reference = measure_fn(reference_theta)
    lower_bounds = jnp.asarray(
        [bounds[key][0] for key in selected_parameters], dtype=jnp.float64
    )
    upper_bounds = jnp.asarray(
        [bounds[key][1] for key in selected_parameters], dtype=jnp.float64
    )
    initial_theta = jnp.clip(
        jnp.asarray([initial_values[key] for key in selected_parameters]),
        lower_bounds,
        upper_bounds,
    )
    learning_rates = jnp.asarray(
        [rates[key] for key in selected_parameters], dtype=initial_theta.dtype
    )
    optimizer = optax.adam(
        learning_rate=learning_rates
    )
    iteration_destination = Path(iteration_dir).expanduser().resolve()
    if resume_from is None:
        start_iteration = 0
        theta = initial_theta
        opt_state = optimizer.init(theta)
        history: list[float] = []
        parameter_history: dict[ParameterKey, list[float]] = {
            key: [] for key in selected_parameters
        }
        best_loss = float("inf")
        best_theta = np.asarray(theta, dtype=float)
        iterations_without_improvement = 0
    else:
        restart = load_restart_state(
            resume_from,
            optimizer=optimizer,
            parameter_keys=selected_parameters,
            measure_keys=keys,
            measure_weights=weight_values,
            learning_rates=learning_rates,
            forward_steps=forward_steps,
            target_iterations=num_steps,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            output_dir=iteration_destination,
        )
        start_iteration = restart.next_iteration
        theta = restart.theta
        opt_state = restart.opt_state
        history = restart.history
        parameter_history = restart.parameter_history
        best_loss = restart.best_loss
        best_theta = restart.best_theta
        iterations_without_improvement = restart.iterations_without_improvement
    stopped_early = False

    print(
        f"reference phi={REF_PHI:.6f} deg, c0={REF_C0:.6f} Pa | "
        f"steps={forward_steps} | "
        f"measures={','.join(keys)} | parameters={','.join(selected_parameters)}"
    )
    print(
        "initial state: "
        f"solid_fraction={INITIAL_SOLID_VOLUME_FRACTION:.3f} | "
        f"bulk_density={BULK_DENSITY:.3f} kg/m^3 | "
        f"lithostatic_K0={LATERAL_STRESS_RATIO:.3f}"
    )
    component_slices = measure_component_slices(keys)
    for key, component_slice, weight in zip(
        keys, component_slices, weight_values, strict=True
    ):
        values = reference[component_slice]
        if key == "terminal_height_profile":
            active = np.flatnonzero(np.asarray(values) >= BULK_HEIGHT_CUTOFF)
            endpoint = (
                ORIGIN[0] + CELL_SIZE * int(active[-1])
                if active.size
                else ORIGIN[0]
            )
            print(
                f"  reference {key}: {values.size} samples | "
                f"bulk_x_max={endpoint:.6f} m | "
                f"loss_weight={float(weight):.6g}"
            )
        else:
            print(
                f"  reference {key}={float(values[0]):.12e} | "
                f"loss_weight={float(weight):.6g}"
            )
    for key in selected_parameters:
        print(f"  constant learning rate {key}={rates[key]:.6e}")
    if resume_from is not None:
        print(
            f"resuming from {Path(resume_from).expanduser().resolve()} | "
            f"next iteration={start_iteration} | best loss={best_loss:.12e}"
        )
    if early_stopping_patience is not None:
        print(
            f"early stopping patience={early_stopping_patience} | "
            f"minimum improvement={early_stopping_min_delta:.3e}"
        )
    if save_iterations:
        print(f"checkpoint frequency={checkpoint_frequency} iterations")

    for step in range(start_iteration, num_steps):
        evaluated_theta = np.asarray(theta, dtype=float)
        parameter_map = dict(zip(selected_parameters, evaluated_theta, strict=True))
        friction_angle = parameter_map.get("friction_angle", REF_PHI)
        cohesion = parameter_map.get("cohesion", REF_C0)
        loss, gradient, measures, sensitivities = loss_and_gradient(
            theta,
            reference,
            measure_keys=keys,
            parameter_keys=selected_parameters,
            num_steps=forward_steps,
            measure_weights=weight_values,
        )
        loss_value = float(loss)
        history.append(loss_value)
        for key, value in parameter_map.items():
            parameter_history[key].append(float(value))
        parameter_text = " | ".join(
            f"{key}={value:.6f}" for key, value in parameter_map.items()
        )
        print(
            f"step={step:03d} | loss={loss_value:.6e} | "
            f"{parameter_text}"
        )
        sensitivity_matrix = jnp.atleast_2d(sensitivities)
        if sensitivity_matrix.shape == (1, len(keys)):
            sensitivity_matrix = sensitivity_matrix.T
        for key, component_slice in zip(keys, component_slices, strict=True):
            values = measures[component_slice]
            sensitivities_for_measure = sensitivity_matrix[component_slice]
            if key == "terminal_height_profile":
                derivatives = " | ".join(
                    f"||d(profile)/d({parameter_key})||="
                    f"{float(jnp.linalg.norm(sensitivities_for_measure[:, index])):.9e}"
                    for index, parameter_key in enumerate(selected_parameters)
                )
                print(
                    f"  {key}: min={float(jnp.min(values)):.9e} | "
                    f"max={float(jnp.max(values)):.9e} | {derivatives}"
                )
            else:
                derivatives = " | ".join(
                    f"d({key})/d({parameter_key})="
                    f"{float(sensitivities_for_measure[0, index]):.9e}"
                    for index, parameter_key in enumerate(selected_parameters)
                )
                print(f"  {key}={float(values[0]):.9e} | {derivatives}")

        if loss_value < best_loss - early_stopping_min_delta:
            best_loss = loss_value
            best_theta = evaluated_theta.copy()
            iterations_without_improvement = 0
        else:
            iterations_without_improvement += 1

        should_stop_early = (
            early_stopping_patience is not None
            and iterations_without_improvement >= early_stopping_patience
        )
        if should_stop_early:
            stopped_early = True
            print(
                f"early stopping at step {step}: no loss improvement greater "
                f"than {early_stopping_min_delta:.3e} for "
                f"{early_stopping_patience} evaluations"
            )
            checkpoint_stop_reason = "early_stopping"
        else:
            updates, opt_state = optimizer.update(gradient, opt_state, theta)
            theta = jnp.clip(
                optax.apply_updates(theta, updates),
                lower_bounds,
                upper_bounds,
            )
            checkpoint_stop_reason = "iteration_limit" if step + 1 == num_steps else ""

        should_checkpoint = save_iterations and (
            (step + 1) % checkpoint_frequency == 0
            or should_stop_early
            or step + 1 == num_steps
        )
        if should_checkpoint:
            checkpoint_path = save_iteration_state(
                step,
                loss=loss,
                friction_angle=friction_angle,
                cohesion=cohesion,
                output_dir=iteration_destination,
                restart_payload=build_restart_payload(
                    next_iteration=step + 1,
                    next_theta=theta,
                    opt_state=opt_state,
                    history=history,
                    parameter_history=parameter_history,
                    best_loss=best_loss,
                    best_theta=best_theta,
                    iterations_without_improvement=iterations_without_improvement,
                    parameter_keys=selected_parameters,
                    measure_keys=keys,
                    measure_weights=weight_values,
                    learning_rates=learning_rates,
                    forward_steps=forward_steps,
                    target_iterations=num_steps,
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_min_delta=early_stopping_min_delta,
                    stop_reason=checkpoint_stop_reason,
                ),
            )
            print(f"  saved iteration state: {checkpoint_path}")
        if should_stop_early:
            break

    theta = jnp.asarray(best_theta, dtype=jnp.float64)
    recovered = {
        key: float(value)
        for key, value in zip(selected_parameters, best_theta, strict=True)
    }
    stop_reason = "early stopping" if stopped_early else "iteration limit"
    print(
        f"best loss={best_loss:.12e} | recovered "
        + ", ".join(f"{key}={value:.9f}" for key, value in recovered.items())
        + f" | stop={stop_reason}"
    )
    if save_plots:
        from projects.collapse.collapse_inverse_forward_ad_plots import (
            FIGURES_DIR,
            create_inverse_plots,
        )

        create_inverse_plots(
            loss_history=history,
            parameter_history=parameter_history,
            reference_parameters=reference_values,
            identified_friction_angle=recovered.get("friction_angle", REF_PHI),
            identified_cohesion=recovered.get("cohesion", REF_C0),
            reference_friction_angle=REF_PHI,
            reference_cohesion=REF_C0,
            measure_keys=keys,
            num_steps=forward_steps,
            output_dir=FIGURES_DIR if figure_dir is None else figure_dir,
        )
    if selected_parameters == DEFAULT_PARAMETER_KEYS:
        return recovered["friction_angle"], history
    return recovered, history


def main() -> None:
    """Command-line entry point for material-parameter inverse analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phi-init", type=float, default=10.0)
    parser.add_argument("--cohesion-init", type=float, default=200.0)
    parser.add_argument("--forward-steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--cohesion-learning-rate",
        type=float,
        default=DEFAULT_COHESION_LEARNING_RATE,
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=PARAMETER_KEYS,
        default=list(DEFAULT_PARAMETER_KEYS),
        help="one or more material parameters to infer",
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
        help="directory for loss, parameter, and optimizer-state checkpoints",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=DEFAULT_CHECKPOINT_FREQUENCY,
        help=(
            "save a restart checkpoint every N completed iterations "
            "and always at termination "
            f"(default: {DEFAULT_CHECKPOINT_FREQUENCY})"
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "exactly resume Adam from a versioned iteration checkpoint; "
            "--iterations remains the total target"
        ),
    )
    parser.add_argument(
        "--no-iteration-data",
        action="store_true",
        help="skip all periodic optimization checkpoints",
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
        parameter_keys=args.parameters,
        forward_steps=args.forward_steps,
        phi_init=args.phi_init,
        cohesion_init=args.cohesion_init,
        learning_rate=args.learning_rate,
        cohesion_learning_rate=args.cohesion_learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_plots=not args.no_plots,
        figure_dir=args.figure_dir,
        save_iterations=not args.no_iteration_data,
        checkpoint_frequency=args.checkpoint_frequency,
        iteration_dir=args.iteration_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
