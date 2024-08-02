from typing import Dict, Tuple

import chex
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import ticker

from ..utils.math_helpers import (
    get_pressure_stack,
    get_q_vm_stack,
    get_scalar_shear_strain_stack,
    get_volumetric_strain_stack,
    phi_to_e_stack,
)


def plot_csl(ax, p_range, M):
    CSL = lambda p: M * p
    csl_range = jnp.array([p_range[0], p_range[1]])
    ax.plot(csl_range, CSL(csl_range), ls=(0, (5, 1)), color="red")
    return ax


def make_plot_set1(
    stress_stack,
    strain_stack,
    volume_fraction_stack,
    internal_variables=(),
    internal_variables_labels=(),
    eps_e_stack=None,
    fig_ax=None,
    start_end_markers=True,
):

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.size": 8})
    if fig_ax is None:
        fig, ax = plt.subplots(2, 3, figsize=(8, 3), dpi=200)
    else:
        fig, ax = fig_ax

    # q-p plot
    q_stack = get_q_vm_stack(stress_stack)
    p_stack = get_pressure_stack(stress_stack)

    (line,) = ax[0, 0].plot(p_stack, q_stack, ls="-", marker=None)

    # start end points

    if start_end_markers:
        ax[0, 0].plot(p_stack[0], q_stack[0], ".", color=line.get_color())
        ax[0, 0].plot(p_stack[-1], q_stack[-1], "x", color=line.get_color())

    ax[0, 0].set_ylabel("$q$ [Pa]")
    ax[0, 0].set_xlabel("$p$ [Pa]")
    ax[0, 0].set_xlim([0, None])

    # q-gamma plot
    gamma_stack = get_scalar_shear_strain_stack(strain_stack)

    ax[0, 1].plot(gamma_stack, q_stack, marker=None)

    # start end points
    if start_end_markers:
        ax[0, 1].plot(gamma_stack[0], q_stack[0], ".", color=line.get_color())
        ax[0, 1].plot(gamma_stack[-1], q_stack[-1], "x", color=line.get_color())

    ax[0, 1].set_xlabel("$\gamma$ [-]")
    ax[0, 1].set_ylabel("$q$ [Pa]")

    if eps_e_stack is not None:
        eps_p_stack = strain_stack - eps_e_stack
    else:
        eps_p_stack = strain_stack

    deps_p_stack = jnp.diff(eps_p_stack, axis=0)

    deps_p_v_stack = get_volumetric_strain_stack(deps_p_stack)

    dgamma_p_stack = get_scalar_shear_strain_stack(deps_p_stack)

    ax[1, 0].plot(dgamma_p_stack, deps_p_v_stack)

    # start end points
    if start_end_markers:
        ax[1, 0].plot(dgamma_p_stack[0], deps_p_v_stack[0], ".", color=line.get_color())
        ax[1, 0].plot(dgamma_p_stack[-1], deps_p_v_stack[-1], "x", color=line.get_color())

    ax[1, 0].set_ylabel("$d\epsilon_v^p$ [-]")
    ax[1, 0].set_xlabel("$d\gamma^p$ [-]")

    eps_v = get_volumetric_strain_stack(strain_stack)
    ax[1, 1].plot(gamma_stack, eps_v)
    # start end points
    if start_end_markers:
        ax[1, 1].plot(gamma_stack[0], eps_v[0], ".", color=line.get_color())
        ax[1, 1].plot(gamma_stack[-1], eps_v[-1], "x", color=line.get_color())
    ax[1, 1].set_xlabel("$d\gamma^p$ [-]")
    ax[1, 1].set_ylabel("$\epsilon_v$ [-]")

    # e log p plot
    e_stack = phi_to_e_stack(volume_fraction_stack)
    ax[1, 2].plot(p_stack, e_stack)

    # start end points
    if start_end_markers:
        ax[1, 2].plot(p_stack[0], e_stack[0], ".", color=line.get_color())
        ax[1, 2].plot(p_stack[-1], e_stack[-1], "x", color=line.get_color())

    ax[1, 2].set_xlabel("ln $p$ [-]")
    ax[1, 2].set_ylabel("e [-]")
    ax[1, 2].set_xscale("log")

    ax[1, 2].xaxis.set_major_locator(ticker.MaxNLocator(2))

    if (internal_variables[0] is not None) & (internal_variables[1] is not None):
        ax[0, 2].plot(internal_variables[0], internal_variables[1])
        if start_end_markers:
            ax[0, 2].plot(
                internal_variables[0][0],
                internal_variables[1][0],
                ".",
                color=line.get_color(),
            )
            ax[0, 2].plot(
                internal_variables[0][-1],
                internal_variables[1][-1],
                "x",
                color=line.get_color(),
            )

        ax[0, 2].set_xlabel(internal_variables_labels[0])
        ax[0, 2].set_ylabel(internal_variables_labels[1])
    return fig, ax, line.get_color()


def setup_figure(
    fig_ax, plot_options, savefig_option, post_plot_options, is_matrix=False
):
    if fig_ax is None:
        if is_matrix:
            fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    if plot_options is None:
        plot_options = {}

    if savefig_option is None:
        savefig_option = {}
    if post_plot_options is None:
        post_plot_options = {}

    return fig, ax, plot_options, savefig_option, post_plot_options


def plot_tensor(
    tensor_stack: chex.Array,
    fig_ax: Tuple = None,
    file: str = None,
    plot_options: Dict = None,
    post_plot_options: Dict = None,
    savefig_option: Dict = None,
    xlabel: str = None,
    ylabel: str = None,
):
    fig, axes, plot_options, savefig_option, post_plot_options = setup_figure(
        fig_ax, plot_options, savefig_option, post_plot_options, is_matrix=True
    )
    num_steps = tensor_stack.shape[0]

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                list(range(num_steps)),
                tensor_stack[:, i, j],
                **plot_options,
            )
            if i == 2:
                if xlabel is not None:
                    axes[i, j].set_xlabel(xlabel)
            if j == 0:
                if ylabel is not None:
                    axes[i, j].set_ylabel(ylabel + f"_{i+1}{j+1}")
    plt.tight_layout()
    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, axes


def plot_q_p(
    stress_stack: chex.Array,
    normalize_stress: jnp.float32 = 1.0e6,
    fig_ax: Tuple = None,
    file: str = None,
    scatter=False,
    plot_options: Dict = None,
    post_plot_options: Dict = None,
    savefig_option: Dict = None,
):
    fig, ax, plot_options, savefig_option, post_plot_options = setup_figure(
        fig_ax, plot_options, savefig_option, post_plot_options
    )

    pressure_stack = get_pressure_stack(stress_stack)
    q_vm_stack = get_q_vm_stack(stress_stack)

    if scatter:
        ax.scatter(
            pressure_stack / normalize_stress,
            q_vm_stack / normalize_stress,
            **plot_options,
        )
    else:
        ax.plot(
            pressure_stack / normalize_stress,
            q_vm_stack / normalize_stress,
            **plot_options,
        )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    if post_plot_options.get("grid") is not None:
        ax.grid(True)

    ax.set_xlabel(f"$p$ {units}")
    ax.set_ylabel(f"$q$ {units}")

    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, ax


def plot_q_gamma(
    stress_stack: chex.Array,
    strain_stack: chex.Array,
    normalize_stress=1.0e6,
    fig_ax: Tuple = None,
    file: Tuple = None,
    plot_options: Dict = None,
    post_plot_options: Dict = None,
    savefig_option: Dict = None,
):
    fig, ax, plot_options, savefig_option, post_plot_options = setup_figure(
        fig_ax, plot_options, savefig_option, post_plot_options
    )

    q_vm_stack = get_q_vm_stack(stress_stack)

    gamma_stack = get_scalar_shear_strain_stack(strain_stack)

    ax.plot(
        gamma_stack,
        q_vm_stack / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    if post_plot_options.get("grid") is not None:
        ax.grid(True)

    ax.set_xlabel("$\gamma$ (-)")
    ax.set_ylabel(f"$q$ {units}")

    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, ax


# def plot_p_gamma(
#     stress_stack: chex.Array,
#     strain_stack: chex.Array,
#     normalize_stress=1.0e6,
#     fig_ax: Tuple = None,
#     file: Tuple = None,
#     plot_options: Dict = None,
#     post_plot_options: Dict = None,
#     savefig_option: Dict = None,
# ):
#     fig, ax, plot_options, savefig_option, post_plot_options = setup_figure(
#         fig_ax, plot_options, savefig_option, post_plot_options
#     )

#     p_list = get_pressure(stress_stack)

#     gamma_stack = get_gamma(strain_stack)

#     ax.plot(
#         gamma_stack,
#         p_list / normalize_stress,
#         **plot_options,
#     )

#     units = "(Pa)"
#     if jnp.isclose(normalize_stress, 1.0e6):
#         units = "(MPa)"

#     if post_plot_options.get("grid") is not None:
#         ax.grid(True)

#     ax.set_xlabel("$\gamma$ (-)")
#     ax.set_ylabel(f"$p$ {units}")
#     if file is not None:
#         fig.savefig(file, **savefig_option)
#         fig.clear()

#     return fig, ax


# def plot_q_dot_gamma(
#     stress_stack: chex.Array,
#     strain_rate_stack: chex.Array,
#     normalize_stress=1.0e6,
#     fig_ax: Tuple = None,
#     file: Tuple = None,
#     plot_options: Dict = None,
#     post_plot_options: Dict = None,
#     savefig_option: Dict = None,
# ):
#     fig, ax, plot_options, savefig_option, post_plot_options = setup_figure(
#         fig_ax, plot_options, savefig_option, post_plot_options
#     )

#     q_vm_stack = get_q_vm(stress_stack)

#     dot_gamma_stack = get_gamma(strain_rate_stack)

#     ax.plot(
#         dot_gamma_stack,
#         q_vm_stack / normalize_stress,
#         **plot_options,
#     )

#     units = "(Pa)"
#     if jnp.isclose(normalize_stress, 1.0e6):
#         units = "(MPa)"

#     if post_plot_options.get("grid") is not None:
#         ax.grid(True)

#     ax.set_xlabel("$\dot \gamma$ (-)")
#     ax.set_ylabel(f"$q$ {units}")
#     if file is not None:
#         fig.savefig(file, **savefig_option)
#         fig.clear()

#     return fig, ax


# def plot_p_dot_gamma(
#     stress_stack: chex.Array,
#     strain_rate_stack: chex.Array,
#     normalize_stress=1.0e6,
#     fig_ax: Tuple = None,
#     file: Tuple = None,
#     plot_options: Dict = None,
#     post_plot_options: Dict = None,
#     savefig_option: Dict = None,
# ):
#     fig, ax, plot_options, savefig_option, post_plot_options = setup_figure(
#         fig_ax, plot_options, savefig_option, post_plot_options
#     )

#     p_list = get_pressure(stress_stack)

#     dot_gamma_stack = get_gamma(strain_rate_stack)

#     ax.plot(
#         dot_gamma_stack,
#         p_list / normalize_stress,
#         **plot_options,
#     )

#     units = "(Pa)"
#     if jnp.isclose(normalize_stress, 1.0e6):
#         units = "(MPa)"

#     if post_plot_options.get("grid") is not None:
#         ax.grid(True)

#     ax.set_xlabel("$\dot \gamma$ (-)")
#     ax.set_ylabel(f"$p$ {units}")
#     if file is not None:
#         fig.savefig(file, **savefig_option)
#         fig.clear()

#     return fig, ax


# def plot_suite(
#     prefix, strain_rate_stack, stress_stack, volume_fraction_stack, dt, fig_ax_stack=None
# ):
#     strain_stack = strain_rate_stack.cumsum(axis=0) * dt

#     plot_options = {"linewidth": 2.0, "ls": "-"}
#     post_plot_options = {"grid": True}

#     fig_ax_stack_new = []

#     if fig_ax_stack is None:
#         fig_ax_stack = [None] * 100  # arbitrary large number of figures

#     fig_ax_stack_new.append(
#         plot_strain_grid(
#             strain_stack,
#             plot_options=plot_options,
#             file=prefix + "_strain_grid.png",
#             fig_ax=fig_ax_stack[0],
#         )
#     )

#     fig_ax_stack_new.append(
#         plot_stress_grid(
#             stress_stack,
#             plot_options=plot_options,
#             file=prefix + "_stress_grid.png",
#             fig_ax=fig_ax_stack[1],
#             normalize_stress=1,
#         )
#     )

#     # fig_ax_stack_new.append(
#     #     plot_strain_rate_grid(
#     #         strain_rate_stack, plot_options=plot_options, file=prefix + "_strain_rate_grid.png", fig_ax=fig_ax_stack[0]
#     #     )
#     # )

#     fig_ax_stack_new.append(
#         plot_q_p(
#             stress_stack,
#             plot_options=plot_options,
#             post_plot_options=post_plot_options,
#             file=prefix + "_q_p.png",
#             fig_ax=fig_ax_stack[2],
#             normalize_stress=1,
#         )
#     )
#     fig_ax_stack_new.append(
#         plot_q_gamma(
#             stress_stack,
#             strain_stack,
#             plot_options=plot_options,
#             post_plot_options=post_plot_options,
#             file=prefix + "_q_gamma.png",
#             fig_ax=fig_ax_stack[3],
#             normalize_stress=1,
#         )
#     )

#     # fig_ax_stack_new.append(
#     #     plot_p_gamma(
#     #         stress_stack,
#     #         strain_stack,
#     #         plot_options=plot_options,
#     #         post_plot_options=post_plot_options,
#     #         file=prefix + "_p_gamma.png",
#     #         fig_ax=fig_ax_stack[4],
#     #     )
#     # )

#     # fig_ax_stack_new.append(
#     #     plot_q_dot_gamma(
#     #         stress_stack,
#     #         strain_rate_stack,
#     #         plot_options=plot_options,
#     #         post_plot_options=post_plot_options,
#     #         file=prefix + "_q_dot_gamma.png",
#     #         fig_ax=fig_ax_stack[5],
#     #     )
#     # )

#     # fig_ax_stack_new.append(
#     #     plot_p_dot_gamma(
#     #         stress_stack,
#     #         strain_rate_stack,
#     #         plot_options=plot_options,
#     #         post_plot_options=post_plot_options,
#     #         file=prefix + "_p_dot_gamma.png",
#     #         fig_ax=fig_ax_stack[6],
#     #     )
#     # )

#     return fig_ax_stack_new
