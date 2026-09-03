"""Persistence utilities for granular-collapse inverse analyses."""

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

CHECKPOINT_SCHEMA_VERSION = 1
ITERATION_FILE_PREFIX = "collapse_inverse_iteration_"


class CheckpointError(ValueError):
    """Raised when an inverse checkpoint cannot be resumed safely."""


@dataclass
class InverseRestartState:
    """Complete optimizer and early-stopping state restored from one iteration."""

    next_iteration: int
    theta: jax.Array
    opt_state: optax.OptState
    history: list[float]
    parameter_history: dict[str, list[float]]
    best_loss: float
    best_theta: np.ndarray
    iterations_without_improvement: int


def atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    """Atomically replace ``path`` with one compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def save_iteration_state(
    iteration: int,
    *,
    loss: float | jax.Array,
    friction_angle: float | jax.Array,
    cohesion: float | jax.Array,
    output_dir: str | Path,
    restart_payload: dict[str, np.ndarray] | None = None,
) -> Path:
    """Save optimization and restart state without simulation field data."""
    destination = Path(output_dir)
    path = destination / f"{ITERATION_FILE_PREFIX}{iteration:04d}.npz"
    payload = {
        "iteration": np.asarray(iteration, dtype=np.int64),
        "loss": np.asarray(float(loss), dtype=np.float64),
        "friction_angle": np.asarray(float(friction_angle), dtype=np.float64),
        "cohesion": np.asarray(float(cohesion), dtype=np.float64),
    }
    if restart_payload is not None:
        overlap = payload.keys() & restart_payload.keys()
        if overlap:
            raise ValueError(f"restart payload duplicates iteration fields: {overlap}")
        payload.update(restart_payload)
    atomic_savez(path, payload)
    return path


def build_restart_payload(
    *,
    next_iteration: int,
    next_theta: jax.Array,
    opt_state: optax.OptState,
    history: Sequence[float],
    parameter_history: dict[str, list[float]],
    best_loss: float,
    best_theta: np.ndarray,
    iterations_without_improvement: int,
    parameter_keys: tuple[str, ...],
    measure_keys: tuple[str, ...],
    measure_weights: jax.Array,
    learning_rates: jax.Array,
    forward_steps: int,
    target_iterations: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    stop_reason: str,
) -> dict[str, np.ndarray]:
    """Serialize all state required to continue Adam exactly."""
    optimizer_leaves, _ = jax.tree_util.tree_flatten(opt_state)
    payload = {
        "checkpoint_schema_version": np.asarray(
            CHECKPOINT_SCHEMA_VERSION, dtype=np.int64
        ),
        "next_iteration": np.asarray(next_iteration, dtype=np.int64),
        "next_theta": np.asarray(next_theta),
        "best_loss": np.asarray(best_loss, dtype=np.float64),
        "best_theta": np.asarray(best_theta, dtype=np.float64),
        "iterations_without_improvement": np.asarray(
            iterations_without_improvement, dtype=np.int64
        ),
        "loss_history": np.asarray(history, dtype=np.float64),
        "parameter_history": np.asarray(
            [parameter_history[key] for key in parameter_keys], dtype=np.float64
        ),
        "parameter_keys": np.asarray(parameter_keys, dtype="U32"),
        "measure_keys": np.asarray(measure_keys, dtype="U32"),
        "measure_weights": np.asarray(measure_weights, dtype=np.float64),
        "learning_rates": np.asarray(learning_rates, dtype=np.float64),
        "forward_steps": np.asarray(forward_steps, dtype=np.int64),
        "target_iterations": np.asarray(target_iterations, dtype=np.int64),
        "early_stopping_patience": np.asarray(
            -1 if early_stopping_patience is None else early_stopping_patience,
            dtype=np.int64,
        ),
        "early_stopping_min_delta": np.asarray(
            early_stopping_min_delta, dtype=np.float64
        ),
        "stop_reason": np.asarray(stop_reason, dtype="U32"),
        "optimizer_leaf_count": np.asarray(len(optimizer_leaves), dtype=np.int64),
    }
    for index, leaf in enumerate(optimizer_leaves):
        payload[f"optimizer_leaf_{index:03d}"] = np.asarray(leaf)
    return payload


def _checkpoint_iteration(path: Path) -> int | None:
    """Extract an iteration index from a checkpoint filename."""
    if path.suffix != ".npz" or not path.stem.startswith(ITERATION_FILE_PREFIX):
        return None
    suffix = path.stem.removeprefix(ITERATION_FILE_PREFIX)
    return int(suffix) if suffix.isdigit() else None


def _validate_no_newer_iterations(output_dir: Path, next_iteration: int) -> None:
    """Prevent a resumed run from silently overwriting later checkpoints."""
    newer = sorted(
        path
        for path in output_dir.glob(f"{ITERATION_FILE_PREFIX}*.npz")
        if (index := _checkpoint_iteration(path)) is not None
        and index >= next_iteration
    )
    if newer:
        raise CheckpointError(
            "resume would overwrite newer iteration checkpoint(s); use a clean "
            f"iteration directory or resume from the latest file: {newer[0]}"
        )


def load_restart_state(
    checkpoint: str | Path,
    *,
    optimizer: optax.GradientTransformation,
    parameter_keys: tuple[str, ...],
    measure_keys: tuple[str, ...],
    measure_weights: jax.Array,
    learning_rates: jax.Array,
    forward_steps: int,
    target_iterations: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    output_dir: Path,
) -> InverseRestartState:
    """Load and validate one exact inverse-analysis restart checkpoint."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise CheckpointError(f"restart checkpoint does not exist: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "iteration",
                "checkpoint_schema_version",
                "next_iteration",
                "next_theta",
                "best_loss",
                "best_theta",
                "iterations_without_improvement",
                "loss_history",
                "parameter_history",
                "parameter_keys",
                "measure_keys",
                "measure_weights",
                "learning_rates",
                "forward_steps",
                "target_iterations",
                "early_stopping_patience",
                "early_stopping_min_delta",
                "stop_reason",
                "optimizer_leaf_count",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise CheckpointError(
                    "legacy or incomplete checkpoint; missing restart field(s): "
                    + ", ".join(missing)
                )
            schema_version = int(archive["checkpoint_schema_version"])
            if schema_version != CHECKPOINT_SCHEMA_VERSION:
                raise CheckpointError(
                    f"unsupported checkpoint schema {schema_version}; expected "
                    f"{CHECKPOINT_SCHEMA_VERSION}"
                )

            stored_parameter_keys = tuple(
                str(value) for value in archive["parameter_keys"]
            )
            stored_measure_keys = tuple(str(value) for value in archive["measure_keys"])
            if stored_parameter_keys != parameter_keys:
                raise CheckpointError(
                    "checkpoint parameter keys do not match this run: "
                    f"{stored_parameter_keys} != {parameter_keys}"
                )
            if stored_measure_keys != measure_keys:
                raise CheckpointError(
                    "checkpoint measure keys do not match this run: "
                    f"{stored_measure_keys} != {measure_keys}"
                )
            if not np.array_equal(
                np.asarray(archive["measure_weights"]), np.asarray(measure_weights)
            ):
                raise CheckpointError("checkpoint measure weights do not match")
            if not np.array_equal(
                np.asarray(archive["learning_rates"]), np.asarray(learning_rates)
            ):
                raise CheckpointError("checkpoint learning rates do not match")
            if int(archive["forward_steps"]) != forward_steps:
                raise CheckpointError("checkpoint forward horizon does not match")
            stored_patience = int(archive["early_stopping_patience"])
            expected_patience = (
                -1 if early_stopping_patience is None else early_stopping_patience
            )
            if stored_patience != expected_patience:
                raise CheckpointError(
                    "checkpoint early-stopping patience does not match"
                )
            if float(archive["early_stopping_min_delta"]) != early_stopping_min_delta:
                raise CheckpointError(
                    "checkpoint early-stopping minimum improvement does not match"
                )

            next_iteration = int(archive["next_iteration"])
            stored_iteration = int(archive["iteration"])
            stored_target_iterations = int(archive["target_iterations"])
            stop_reason = str(archive["stop_reason"])
            if next_iteration <= 0 or stored_iteration != next_iteration - 1:
                raise CheckpointError("checkpoint iteration indices are inconsistent")
            if stored_target_iterations < next_iteration:
                raise CheckpointError(
                    "checkpoint target iteration count is inconsistent"
                )
            if stop_reason not in ("", "iteration_limit", "early_stopping"):
                raise CheckpointError("checkpoint stop reason is invalid")
            if stop_reason == "" and next_iteration >= stored_target_iterations:
                raise CheckpointError("checkpoint completion state is inconsistent")
            if stop_reason == "iteration_limit" and (
                next_iteration != stored_target_iterations
            ):
                raise CheckpointError(
                    "checkpoint iteration-limit state is inconsistent"
                )
            if stop_reason == "early_stopping":
                raise CheckpointError("checkpoint already stopped early")
            if next_iteration >= target_iterations:
                raise CheckpointError(
                    "checkpoint already reached the requested total iteration count"
                )
            _validate_no_newer_iterations(output_dir, next_iteration)

            theta = jnp.asarray(archive["next_theta"], dtype=jnp.float64)
            expected_shape = (len(parameter_keys),)
            if theta.shape != expected_shape:
                raise CheckpointError(
                    f"checkpoint parameter shape {theta.shape} != {expected_shape}"
                )
            if not bool(jnp.all(jnp.isfinite(theta))):
                raise CheckpointError("checkpoint parameters are nonfinite")
            template_state = optimizer.init(theta)
            template_leaves, tree_definition = jax.tree_util.tree_flatten(
                template_state
            )
            leaf_count = int(archive["optimizer_leaf_count"])
            if leaf_count != len(template_leaves):
                raise CheckpointError(
                    "checkpoint Adam state is incompatible with the current optimizer"
                )
            restored_leaves = []
            for index, template_leaf in enumerate(template_leaves):
                key = f"optimizer_leaf_{index:03d}"
                if key not in archive.files:
                    raise CheckpointError(f"checkpoint is missing {key}")
                saved_leaf = np.asarray(archive[key])
                template_array = np.asarray(template_leaf)
                if (
                    saved_leaf.shape != template_array.shape
                    or saved_leaf.dtype != template_array.dtype
                ):
                    raise CheckpointError(
                        f"checkpoint Adam leaf {index} has incompatible shape or dtype"
                    )
                if np.issubdtype(saved_leaf.dtype, np.inexact) and not np.all(
                    np.isfinite(saved_leaf)
                ):
                    raise CheckpointError(
                        f"checkpoint Adam leaf {index} contains nonfinite values"
                    )
                restored_leaves.append(jnp.asarray(saved_leaf))
            opt_state = jax.tree_util.tree_unflatten(tree_definition, restored_leaves)

            history = np.asarray(archive["loss_history"], dtype=float)
            stored_parameter_history = np.asarray(
                archive["parameter_history"], dtype=float
            )
            if history.shape != (next_iteration,):
                raise CheckpointError("checkpoint loss history length is inconsistent")
            if stored_parameter_history.shape != (
                len(parameter_keys),
                next_iteration,
            ):
                raise CheckpointError(
                    "checkpoint parameter history shape is inconsistent"
                )
            best_theta = np.asarray(archive["best_theta"], dtype=float)
            if best_theta.shape != expected_shape:
                raise CheckpointError("checkpoint best-parameter shape is inconsistent")
            best_loss = float(archive["best_loss"])
            no_improvement = int(archive["iterations_without_improvement"])
            if (
                not np.all(np.isfinite(history))
                or not np.all(np.isfinite(stored_parameter_history))
                or not np.all(np.isfinite(best_theta))
                or not np.isfinite(best_loss)
            ):
                raise CheckpointError("checkpoint optimization history is nonfinite")
            if no_improvement < 0 or no_improvement > next_iteration:
                raise CheckpointError(
                    "checkpoint early-stopping counter is inconsistent"
                )
            return InverseRestartState(
                next_iteration=next_iteration,
                theta=theta,
                opt_state=opt_state,
                history=history.tolist(),
                parameter_history={
                    key: stored_parameter_history[index].tolist()
                    for index, key in enumerate(parameter_keys)
                },
                best_loss=best_loss,
                best_theta=best_theta,
                iterations_without_improvement=no_improvement,
            )
    except CheckpointError:
        raise
    except Exception as error:
        message = f"could not read restart checkpoint {path}: {error}"
        raise CheckpointError(message) from error
