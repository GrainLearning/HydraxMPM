import matplotlib.pyplot as plt
import jax.numpy as jnp
import chex

from typing import Tuple, Dict
from ..utils.math_helpers import get_gamma, get_pressure, get_q_vm


def setup_figure(fig_ax, plot_options, savefig_option, post_plot_options, is_matrix=False):
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


def plot_strain_grid(
    strain_stack: chex.Array,
    fig_ax: Tuple = None,
    file: str = None,
    plot_options: Dict = None,
    post_plot_options: Dict = None,
    savefig_option: Dict = None,
):
    fig, axes, plot_options, savefig_option, post_plot_options = setup_figure(
        fig_ax, plot_options, savefig_option, post_plot_options, is_matrix=True
    )
    num_steps = strain_stack.shape[0]

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                list(range(num_steps)),
                strain_stack[:, i, j],
                **plot_options,
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(f"$\\varepsilon_{{{i+1}{j+1}}}$")

    plt.tight_layout()
    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, axes


def plot_stress_grid(
    stress_stack: chex.Array,
    normalize_stress: jnp.float32 = 1.0e6,
    fig_ax: Tuple = None,
    file: str = None,
    plot_options: Dict = None,
    post_plot_options: Dict = None,
    savefig_option: Dict = None,
):
    fig, axes, plot_options, savefig_option, post_plot_options = setup_figure(
        fig_ax, plot_options, savefig_option, post_plot_options, is_matrix=True
    )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    num_steps = stress_stack.shape[0]

    for i in range(3):
        for j in range(3):
            axes[i, j].plot(
                list(range(num_steps)),
                stress_stack[:, i, j] / normalize_stress,
                **plot_options,
            )

            if i == 2:
                axes[i, j].set_xlabel("step")
            if j == 0:
                axes[i, j].set_ylabel(rf"$\sigma_{{{i+1}{j+1}}}$ {units}")

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

    pressure_list = get_pressure(stress_stack)
    q_vm_list = get_q_vm(stress_stack)

    if scatter:
        ax.scatter(
            pressure_list / normalize_stress,
            q_vm_list / normalize_stress,
            **plot_options,
        )
    else:
        ax.plot(
            pressure_list / normalize_stress,
            q_vm_list / normalize_stress,
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

    q_vm_list = get_q_vm(stress_stack)

    gamma_list = get_gamma(strain_stack)

    ax.plot(
        gamma_list,
        q_vm_list / normalize_stress,
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


def plot_p_gamma(
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

    p_list = get_pressure(stress_stack)

    gamma_list = get_gamma(strain_stack)

    ax.plot(
        gamma_list,
        p_list / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    if post_plot_options.get("grid") is not None:
        ax.grid(True)

    ax.set_xlabel("$\gamma$ (-)")
    ax.set_ylabel(f"$p$ {units}")
    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, ax


def plot_q_dot_gamma(
    stress_stack: chex.Array,
    strain_rate_stack: chex.Array,
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

    q_vm_list = get_q_vm(stress_stack)

    dot_gamma_list = get_gamma(strain_rate_stack)

    ax.plot(
        dot_gamma_list,
        q_vm_list / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    if post_plot_options.get("grid") is not None:
        ax.grid(True)

    ax.set_xlabel("$\dot \gamma$ (-)")
    ax.set_ylabel(f"$q$ {units}")
    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, ax


def plot_p_dot_gamma(
    stress_stack: chex.Array,
    strain_rate_stack: chex.Array,
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

    p_list = get_pressure(stress_stack)

    dot_gamma_list = get_gamma(strain_rate_stack)

    ax.plot(
        dot_gamma_list,
        p_list / normalize_stress,
        **plot_options,
    )

    units = "(Pa)"
    if jnp.isclose(normalize_stress, 1.0e6):
        units = "(MPa)"

    if post_plot_options.get("grid") is not None:
        ax.grid(True)

    ax.set_xlabel("$\dot \gamma$ (-)")
    ax.set_ylabel(f"$p$ {units}")
    if file is not None:
        fig.savefig(file, **savefig_option)
        fig.clear()

    return fig, ax


def plot_suite(prefix, strain_rate_stack, stress_stack, volume_fraction_stack, material_stack, fig_ax_stack=None):
    strain_stack = strain_rate_stack.cumsum(axis=0)

    plot_options = {"linewidth": 2.0, "ls": "-"}
    post_plot_options = {"grid": True}

    fig_ax_stack_new = []

    if fig_ax_stack is None:
        fig_ax_stack = [None] * 100  # arbitrary large number of figures

    fig_ax_stack_new.append(
        plot_strain_grid(
            strain_stack, plot_options=plot_options, file=prefix + "_strain_grid.png", fig_ax=fig_ax_stack[0]
        )
    )

    fig_ax_stack_new.append(
        plot_stress_grid(
            stress_stack, plot_options=plot_options, file=prefix + "_stress_grid.png", fig_ax=fig_ax_stack[1]
        )
    )

    fig_ax_stack_new.append(
        plot_q_p(
            stress_stack,
            plot_options=plot_options,
            post_plot_options=post_plot_options,
            file=prefix + "_q_p.png",
            fig_ax=fig_ax_stack[2],
        )
    )
    fig_ax_stack_new.append(
        plot_q_gamma(
            stress_stack,
            strain_stack,
            plot_options=plot_options,
            post_plot_options=post_plot_options,
            file=prefix + "_q_gamma.png",
            fig_ax=fig_ax_stack[3],
        )
    )

    fig_ax_stack_new.append(
        plot_p_gamma(
            stress_stack,
            strain_stack,
            plot_options=plot_options,
            post_plot_options=post_plot_options,
            file=prefix + "_p_gamma.png",
            fig_ax=fig_ax_stack[4],
        )
    )

    fig_ax_stack_new.append(
        plot_q_dot_gamma(
            stress_stack,
            strain_rate_stack,
            plot_options=plot_options,
            post_plot_options=post_plot_options,
            file=prefix + "_q_dot_gamma.png",
            fig_ax=fig_ax_stack[5],
        )
    )

    fig_ax_stack_new.append(
        plot_p_dot_gamma(
            stress_stack,
            strain_rate_stack,
            plot_options=plot_options,
            post_plot_options=post_plot_options,
            file=prefix + "_p_dot_gamma.png",
            fig_ax=fig_ax_stack[6],
        )
    )

    return fig_ax_stack_new
