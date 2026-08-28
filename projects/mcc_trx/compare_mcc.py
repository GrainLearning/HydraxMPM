"""Compare exported HydraxMPM MCC predictions with analytical references."""
# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import runpy
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from mcc_ad_tangents import (
    compute_tangent_error_history,
    compute_tangent_sensitivities,
)
from mcc_benchmark_config import (
    BASE_PATH,
    CONFINE,
    DATA_PATH,
    M,
    NUM_STEPS,
)

A4_FIGSIZE = (11.69 * 2 / 3, 8.27 / 3)
PANEL_FIGSIZE = (A4_FIGSIZE[0] / 2, A4_FIGSIZE[1])


def configure_plotting() -> None:
    rcParams["mathtext.fontset"] = "cm"
    rcParams["font.family"] = "serif"
    rcParams["font.size"] = 14
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(BASE_PATH / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


@dataclass
class AnalyticalResult:
    axial_strain: np.ndarray
    radial_strain: np.ndarray
    pressure: np.ndarray
    deviatoric_stress: np.ndarray
    volumetric_strain: np.ndarray
    specific_volume: np.ndarray
    preconsolidation_pressure: np.ndarray
    stress_strain_tangent: np.ndarray


def analytical_result(mode: str) -> AnalyticalResult:
    argv = sys.argv
    try:
        sys.argv = [str(BASE_PATH / "mcc_close_form_D_ep.py")]
        if mode == "undrained":
            sys.argv.append("--undrained")
        data = runpy.run_path(sys.argv[0])
    finally:
        sys.argv = argv

    result = AnalyticalResult(
        axial_strain=data["axial_strain"],
        radial_strain=data["radial_strain"],
        pressure=data["p"] * 1_000,
        deviatoric_stress=data["q"] * 1_000,
        volumetric_strain=data["ev"],
        specific_volume=1 + data["void_ratio_total"],
        preconsolidation_pressure=data["pc_surface"] * 1_000,
        stress_strain_tangent=data["tangent_history"],
    )
    np.savetxt(
        DATA_PATH / f"mcc_close_form_D_ep_{mode}.csv",
        np.column_stack(
            (
                result.axial_strain,
                result.radial_strain,
                result.pressure,
                result.deviatoric_stress,
                result.volumetric_strain,
                result.specific_volume,
                result.preconsolidation_pressure,
                result.stress_strain_tangent.reshape(-1, 4),
            )
        ),
        delimiter=",",
        header=(
            "axial_strain,radial_strain,pressure_Pa,q_Pa,volumetric_strain,"
            "specific_volume,pc_Pa,dp_daxial_Pa,dp_dradial_Pa,"
            "dq_daxial_Pa,dq_dradial_Pa"
        ),
        comments="",
    )
    return result


def generate_benchmark_data() -> dict[str, AnalyticalResult]:
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    return {mode: analytical_result(mode) for mode in ("drained", "undrained")}


def load_prediction(mode: str) -> np.ndarray:
    path = DATA_PATH / f"mcc_numerical_{mode}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run projects/mcc_trx/mcc.py first; missing {path}")
    prediction = np.genfromtxt(path, delimiter=",", names=True)
    if prediction.shape[0] != NUM_STEPS + 1:
        raise ValueError(f"Expected {NUM_STEPS + 1} rows in {path}")
    return prediction


def comparison_metrics(
    analytical: AnalyticalResult,
    prediction: np.ndarray,
    *,
    is_undrained: bool,
) -> dict[str, float]:
    p_error = prediction["pressure_Pa"] - analytical.pressure
    q_error = prediction["q_Pa"] - analytical.deviatoric_stress
    ev_error = prediction["volumetric_strain"] - analytical.volumetric_strain
    pc_error = prediction["pc_Pa"] - analytical.preconsolidation_pressure
    return {
        "p_nrmse_percent": 100.0 * np.sqrt(np.mean(p_error**2)) / CONFINE,
        "q_nrmse_percent": 100.0 * np.sqrt(np.mean(q_error**2)) / (M * CONFINE),
        "eps_v_rmse": np.sqrt(np.mean(ev_error**2)),
        "pc_nrmse_percent": 100.0 * np.sqrt(np.mean(pc_error**2)) / CONFINE,
        "final_p_error_percent": 100.0 * p_error[-1] / analytical.pressure[-1],
        "final_q_error_percent": 100.0 * q_error[-1] / analytical.deviatoric_stress[-1],
        "max_drained_confinement_error_Pa": float(
            np.max(
                np.abs(
                    prediction["pressure_Pa"]
                    - prediction["q_Pa"] / 3.0
                    - CONFINE
                )
            )
        )
        if not is_undrained
        else np.nan,
        "max_undrained_abs_eps_v": float(
            np.max(np.abs(prediction["volumetric_strain"]))
        )
        if is_undrained
        else np.nan,
    }


def write_rows(path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_error_histories(histories) -> None:
    fieldnames = [
        "mode",
        "axial_strain",
        "dp_daxial_error_percent",
        "dp_dradial_error_percent",
        "dq_daxial_error_percent",
        "dq_dradial_error_percent",
        "tangent_norm_error_percent",
    ]
    rows = []
    for mode in ("drained", "undrained"):
        history = histories[mode]
        for index, strain in enumerate(history["axial_strain"]):
            rows.append(
                {
                    "mode": mode,
                    "axial_strain": strain,
                    **{key: history[key][index] for key in fieldnames[2:]},
                }
            )
    write_rows(DATA_PATH / "stress_strain_tangent_error_history.csv", rows)


def plot_comparison(analytical, predictions) -> None:
    response_fig, strain_axis = plt.subplots(
        figsize=(A4_FIGSIZE[0] / 1.5, A4_FIGSIZE[1]), constrained_layout=True
    )
    volume_axis = strain_axis.twinx()
    path_fig, path_axis = plt.subplots(
        figsize=(A4_FIGSIZE[0] / 1.8, A4_FIGSIZE[1]), constrained_layout=True)
    linestyles = {"drained": "-", "undrained": "--"}
    markers = {"drained": "o", "undrained": "s"}
    for mode in ("drained", "undrained"):
        ref = analytical[mode]
        prediction = predictions[mode]
        marker_steps = np.linspace(0, len(prediction) - 1, 30, dtype=int)
        strain_axis.plot(
            ref.axial_strain,
            ref.deviatoric_stress / ref.pressure,
            color="black",
            linestyle=linestyles[mode],
            linewidth=1.2,
            label=rf"{mode} $q/p$ (ref.)",
        )
        strain_axis.plot(
            prediction["axial_strain"][marker_steps],
            (prediction["q_Pa"] / prediction["pressure_Pa"])[marker_steps],
            linestyle="none",
            marker=markers[mode],
            ms=4.0,
            markerfacecolor="none",
            color="black",
            label=rf"{mode} $q/p$ (num.)",
        )
        volume_axis.plot(
            ref.axial_strain,
            ref.volumetric_strain,
            color="#0072B2",
            linewidth=1.0,
            linestyle=linestyles[mode],
            label=rf"{mode} $\varepsilon_v$ (ref.)",
        )
        volume_axis.plot(
            prediction["axial_strain"][marker_steps],
            prediction["volumetric_strain"][marker_steps],
            linestyle="none",
            marker=markers[mode],
            ms=4.0,
            markerfacecolor="none",
            color="#0072B2",
            label=rf"{mode} $\varepsilon_v$ (num.)",
        )
        path_axis.plot(
            ref.pressure / 1_000.0,
            ref.deviatoric_stress / 1_000.0,
            color='black',
            linestyle=linestyles[mode],
            linewidth=1.2,
            label=f"{mode} (ref.)",
        )
        path_axis.plot(
            prediction["pressure_Pa"][marker_steps] / 1_000.0,
            prediction["q_Pa"][marker_steps] / 1_000.0,
            linestyle="none",
            marker=markers[mode],
            ms=4.0,
            markerfacecolor="none",
            color="black",
            label=f"{mode} (num.)",
        )
    p_line = np.linspace(0.0, 16.0, 100)
    path_axis.plot(p_line, M * p_line, color="darkred", lw=1.0, label=r"CSL $q=Mp$")
    strain_axis.set_ylim(ymax=1.0)
    strain_axis.set(xlabel=r"$\varepsilon_a$ [-]", ylabel=r"$q/p$ [-]")
    volume_axis.set_ylabel(r"$\varepsilon_v$ [-]")
    volume_axis.set_ylim(ymax=0.25)
    volume_axis.yaxis.label.set_color("#0072B2")
    volume_axis.tick_params(axis="y", colors="#0072B2")
    volume_axis.spines["right"].set_color("#0072B2")
    volume_axis.grid(False)
    path_axis.set(
        xlabel=r"$p$ [kPa]",
        ylabel=r"$q$ [kPa]",
    )
    strain_handles, strain_labels = strain_axis.get_legend_handles_labels()
    volume_handles, volume_labels = volume_axis.get_legend_handles_labels()
    response_fig.legend(
        strain_handles + volume_handles,
        strain_labels + volume_labels,
        frameon=False,
        ncol=1,
        loc="outside right center",
    )
    path_axis.legend(
        frameon=False,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    save_figure(response_fig, "mcc_triax_response.png")
    save_figure(path_fig, "mcc_stress_path.png")


def plot_heatmaps(rows) -> None:
    pairs = (
        ("ad_dp_daxial_Pa", "analytical_dp_daxial_Pa"),
        ("ad_dp_dradial_Pa", "analytical_dp_dradial_Pa"),
        ("ad_dq_daxial_Pa", "analytical_dq_daxial_Pa"),
        ("ad_dq_dradial_Pa", "analytical_dq_dradial_Pa"),
    )
    labels = (
        r"$\partial p/\partial\varepsilon_a$",
        r"$\partial p/\partial\varepsilon_r$",
        r"$\partial q/\partial\varepsilon_a$",
        r"$\partial q/\partial\varepsilon_r$",
    )
    errors_by_mode = {}
    for mode in ("drained", "undrained"):
        mode_rows = [row for row in rows if row["mode"] == mode]
        errors = np.empty((4, len(mode_rows)))
        for column, row in enumerate(mode_rows):
            analytical_tangent = np.array(
                [
                    [row[pairs[0][1]], row[pairs[1][1]]],
                    [row[pairs[2][1]], row[pairs[3][1]]],
                ]
            )
            norm = np.linalg.norm(analytical_tangent)
            for component, (ad_key, ref_key) in enumerate(pairs):
                errors[component, column] = 100.0 * (row[ad_key] - row[ref_key]) / norm
        errors_by_mode[mode] = errors
    limit = max(np.max(np.abs(errors)) for errors in errors_by_mode.values())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, mode in zip(axes, ("drained", "undrained"), strict=True):
        errors = errors_by_mode[mode]
        image = axis.imshow(
            errors,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
        )
        axis.set_title(mode.capitalize())
        axis.set_xlabel(r"Axial strain $\varepsilon_a$")
        axis.set_xticks(np.arange(4), ["1%", "5%", "10%", "20%"])
        axis.set_yticks(np.arange(4), labels)
        for row_index, column_index in np.ndindex(errors.shape):
            value = errors[row_index, column_index]
            color = "white" if abs(value) > 0.55 * limit else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=color,
            )
    colorbar = fig.colorbar(image, ax=axes, shrink=0.9)
    colorbar.set_label(
        r"Signed component error $100(D^{AD}_{ij}-D^{ref}_{ij})/\|D^{ref}\|_F$ [%]"
    )
    fig.savefig(BASE_PATH / "mcc_tangent_error_heatmaps.png", dpi=200)
    plt.close(fig)


def plot_error_evolution(histories) -> None:
    curves = (
        ("dp_daxial_error_percent", r"$\partial p/\partial\varepsilon_a$"),
        ("dp_dradial_error_percent", r"$\partial p/\partial\varepsilon_r$"),
        ("dq_daxial_error_percent", r"$\partial q/\partial\varepsilon_a$"),
        ("dq_dradial_error_percent", r"$\partial q/\partial\varepsilon_r$"),
    )
    for mode in ("drained", "undrained"):
        fig, axis = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
        history = histories[mode]
        for key, label in curves:
            error = np.maximum(history[key], np.finfo(float).tiny)
            axis.plot(history["axial_strain"], error, lw=1.0, label=label)
        axis.set_yscale("log")
        axis.set_xlabel(r"$\varepsilon_a$ [-]")
        axis.set_ylabel(r" $\|(D^\mathrm{AD}-D^\mathrm{ref})\|/D^\mathrm{ref}$ [%]")
        axis.legend(frameon=False, ncol=2, loc="best")
        save_figure(fig, f"mcc_tangent_error_{mode}.png")


def main() -> None:
    configure_plotting()
    analytical = generate_benchmark_data()
    predictions = {mode: load_prediction(mode) for mode in ("drained", "undrained")}
    metrics = {
        mode: comparison_metrics(
            analytical[mode],
            predictions[mode],
            is_undrained=(mode == "undrained"),
        )
        for mode in ("drained", "undrained")
    }
    write_rows(
        DATA_PATH / "comparison_summary.csv",
        [{"mode": mode, **values} for mode, values in metrics.items()],
    )
    plot_comparison(analytical, predictions)

    sensitivity_rows = [
        row
        for mode in ("drained", "undrained")
        for row in compute_tangent_sensitivities(
            mode, predictions[mode], analytical[mode]
        )
    ]
    write_rows(DATA_PATH / "stress_strain_tangent_sensitivities.csv", sensitivity_rows)
    plot_heatmaps(sensitivity_rows)

    histories = {
        mode: compute_tangent_error_history(mode, predictions[mode], analytical[mode])
        for mode in ("drained", "undrained")
    }
    write_error_histories(histories)
    plot_error_evolution(histories)

    print("mode,p_NRMSE_percent,q_NRMSE_percent,eps_v_RMSE,final_q_error_percent")
    for mode, values in metrics.items():
        print(
            f"{mode},{values['p_nrmse_percent']:.6f},"
            f"{values['q_nrmse_percent']:.6f},{values['eps_v_rmse']:.6e},"
            f"{values['final_q_error_percent']:.6f}"
        )


if __name__ == "__main__":
    main()
