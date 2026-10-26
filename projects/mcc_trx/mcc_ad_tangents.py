"""Automatic-differentiation tangent evaluation for the MCC benchmark."""
# ruff: noqa: E402, I001

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import hydraxmpm as hdx
from mcc_benchmark_config import (
    AXIAL_RATE,
    CONFINE,
    DT,
    KAP,
    LAM,
    M,
    N,
    NUM_STEPS,
    NU,
    P_REF,
    RHO_P,
    SENSITIVITY_AXIAL_STRAINS,
    SENSITIVITY_PROBE_SCALE,
)

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def create_mcc() -> hdx.ModifiedCamClay:
    return hdx.ModifiedCamClay(
        nu=NU,
        M=M,
        lam=LAM,
        kap=KAP,
        N=N,
        p_ref=P_REF,
        rho_p=RHO_P,
    )


def stress_history(prediction: np.ndarray) -> np.ndarray:
    stress = np.zeros((prediction.shape[0], 3, 3))
    stress[:, 0, 0] = prediction["stress_xx_Pa"]
    stress[:, 1, 1] = prediction["stress_yy_Pa"]
    stress[:, 2, 2] = prediction["stress_zz_Pa"]
    return stress


def elastic_strain_history(prediction: np.ndarray) -> np.ndarray:
    elastic_strain = np.zeros((prediction.shape[0], 3, 3))
    elastic_strain[:, 0, 0] = prediction["eps_e_xx"]
    elastic_strain[:, 1, 1] = prediction["eps_e_yy"]
    elastic_strain[:, 2, 2] = prediction["eps_e_zz"]
    return elastic_strain


def local_stress_response(
    mcc,
    increment,
    eps_e_previous,
    stress_previous,
    pc_previous,
    specific_volume,
):
    axial_increment, radial_increment = increment
    L = jnp.zeros((3, 3), dtype=jnp.float64)
    L = L.at[0, 0].set(radial_increment / DT)
    L = L.at[1, 1].set(radial_increment / DT)
    L = L.at[2, 2].set(axial_increment / DT)
    stress_next, _, _ = mcc._update_stress(
        L,
        eps_e_previous,
        stress_previous,
        pc_previous,
        CONFINE * jnp.eye(3),
        specific_volume,
        DT,
    )
    pressure = jnp.trace(stress_next) / 3.0
    deviatoric = stress_next - pressure * jnp.eye(3)
    q = jnp.sqrt(1.5 * jnp.sum(deviatoric * deviatoric))
    return jnp.array([pressure, q])


def continuation_increments(prediction: np.ndarray, state_steps: np.ndarray):
    next_rows = np.minimum(state_steps + 1, NUM_STEPS)
    return np.column_stack(
        (
            prediction["axial_increment"][next_rows],
            prediction["radial_increment"][next_rows],
        )
    ) * SENSITIVITY_PROBE_SCALE


def evaluate_ad_tangents(
    prediction: np.ndarray, state_steps: np.ndarray
) -> np.ndarray:
    increments = continuation_increments(prediction, state_steps)
    tangent = jax.jacfwd(local_stress_response, argnums=1)
    batched_tangent = jax.jit(jax.vmap(tangent, in_axes=(None, 0, 0, 0, 0, 0)))
    return np.asarray(
        batched_tangent(
            create_mcc(),
            jnp.asarray(increments),
            jnp.asarray(elastic_strain_history(prediction)[state_steps]),
            jnp.asarray(stress_history(prediction)[state_steps]),
            jnp.asarray(prediction["pc_Pa"][state_steps]),
            jnp.asarray(prediction["specific_volume"][state_steps]),
        )
    )


def compute_tangent_sensitivities(
    mode: str,
    prediction: np.ndarray,
    analytical,
) -> list[dict[str, float | str]]:
    steps = np.rint(
        np.asarray(SENSITIVITY_AXIAL_STRAINS) / (AXIAL_RATE * DT)
    ).astype(int)
    ad_tangents = evaluate_ad_tangents(prediction, steps)
    rows = []
    for step, ad_tangent in zip(steps, ad_tangents, strict=True):
        reference = analytical.stress_strain_tangent[step]
        relative_error = np.linalg.norm(ad_tangent - reference) / np.linalg.norm(
            reference
        )
        rows.append(
            {
                "mode": mode,
                "axial_strain": prediction["axial_strain"][step],
                "radial_strain": prediction["radial_strain"][step],
                "pressure_Pa": prediction["pressure_Pa"][step],
                "q_Pa": prediction["q_Pa"][step],
                "analytical_pressure_Pa": analytical.pressure[step],
                "analytical_q_Pa": analytical.deviatoric_stress[step],
                "probe_increment_scale": SENSITIVITY_PROBE_SCALE,
                "ad_dp_daxial_Pa": ad_tangent[0, 0],
                "ad_dp_dradial_Pa": ad_tangent[0, 1],
                "ad_dq_daxial_Pa": ad_tangent[1, 0],
                "ad_dq_dradial_Pa": ad_tangent[1, 1],
                "analytical_dp_daxial_Pa": reference[0, 0],
                "analytical_dp_dradial_Pa": reference[0, 1],
                "analytical_dq_daxial_Pa": reference[1, 0],
                "analytical_dq_dradial_Pa": reference[1, 1],
                "tangent_relative_error_percent": 100.0 * relative_error,
            }
        )
    return rows


def compute_tangent_error_history(
    mode: str,
    prediction: np.ndarray,
    analytical,
) -> dict[str, np.ndarray | str]:
    steps = np.arange(1, NUM_STEPS + 1)
    ad_tangent = evaluate_ad_tangents(prediction, steps)
    reference = analytical.stress_strain_tangent[steps]
    reference_norm = np.linalg.norm(reference, axis=(1, 2))
    component_error = (
        100.0
        * np.abs(ad_tangent - reference)
        / reference_norm[:, np.newaxis, np.newaxis]
    )
    total_error = (
        100.0
        * np.linalg.norm(ad_tangent - reference, axis=(1, 2))
        / reference_norm
    )
    return {
        "mode": mode,
        "axial_strain": prediction["axial_strain"][steps],
        "dp_daxial_error_percent": component_error[:, 0, 0],
        "dp_dradial_error_percent": component_error[:, 0, 1],
        "dq_daxial_error_percent": component_error[:, 1, 0],
        "dq_dradial_error_percent": component_error[:, 1, 1],
        "tangent_norm_error_percent": total_error,
    }
