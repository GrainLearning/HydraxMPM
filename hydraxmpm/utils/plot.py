# Copyright (c) 2024, Retiefasuarus
# SPDX-License-Identifier: BSD-3-Clause
#
# Part of HydraxMPM: https://github.com/GrainLearning/HydraxMPM

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.ticker import (
    LogLocator,
    ScalarFormatter,
    FormatStrFormatter,
    NullFormatter,
    StrMethodFormatter,
)

from matplotlib import ticker


def make_plot(
    ax,
    x,
    y,
    start_end_markers=True,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xlogscale=False,
    ylogscale=False,
    label=None,
    start_end_markersize=None,
    **kwargs,
):
    out = ax.plot(x, y, label=label, **kwargs)
    (line,) = out
    if start_end_markers:
        ax.plot(
            x[0], y[0], ".", color=line.get_color(), markersize=start_end_markersize
        )
        ax.plot(
            x[-1], y[-1], "*", color=line.get_color(), markersize=start_end_markersize
        )

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    def format_func(value, tick_number):
        return "%g" % (value)

    # ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    if xlogscale:
        ax.set_xscale("log")
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%s"))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
    if ylogscale:
        ax.set_yscale("log")
        # ax.semilogy(range(100))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        # ax.ticklabel_format(style="plain", axis="y")
        # ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    return out


def plot_set1(
    p_stack,
    q_vm_stack,
    gamma_stack,
    dgammadt_stack,
    specific_volume_stack,
    inertial_number_stack,
    fig_ax=None,
    **kwargs,
):
    """

    ax_lim = [[xmin,xmax],[ymin,ymax]]

    Args:
        p_stack: _description_
        q_vm_stack: _description_
        gamma_stack: _description_
        dgammadt_stack: _description_
        specific_volume_stack: _description_
        inertial_number_stack: _description_
        fig_ax: _description_. Defaults to None.

    Returns:
        _description_
    """
    if fig_ax is None:
        fig, ax = plt.subplots(
            ncols=3, nrows=2, figsize=(7.5 * 0.9, 4.5 * 0.9), dpi=300
        )
    else:
        fig, ax = fig_ax

    q_p_stack = q_vm_stack / p_stack

    ls = kwargs.get("ls", "-")
    color = kwargs.get("color", None)
    label = kwargs.get("label", None)
    make_plot(
        ax.flat[0],
        p_stack,
        q_vm_stack,
        xlabel="$p$ [Pa]",
        ylabel="$q$ [Pa]",
        ls=ls,
        color=color,
        label=label,
    )
    # ax.flat[0].set_xlim((0, None), auto=True)
    # ax.flat[0].set_ylim((0, None), auto=True)
    # ax.flat[0].margins(0.1, 0.15)
    make_plot(
        ax.flat[1],
        gamma_stack,
        q_vm_stack,
        xlabel="$\\gamma$ [-]",
        ylabel="$q$ [Pa]",
        ls=ls,
        color=color,
    )

    make_plot(
        ax.flat[2],
        specific_volume_stack,
        q_p_stack,
        ylabel="$q/p$ [-]",
        xlabel="$v=\\phi^{-1}$ [-]",
        ls=ls,
        color=color,
    )

    make_plot(
        ax.flat[3],
        p_stack,
        specific_volume_stack,
        xlogscale=True,
        ylogscale=True,
        xlabel="$p$ [Pa] (log-scale)",
        ylabel="$v=\\phi^{-1}$ [-] (log-scale)",
        ls=ls,
        color=color,
    )
    make_plot(
        ax.flat[4],
        gamma_stack,
        q_p_stack,
        xlabel="$\\gamma$ [-]",
        ylabel="$q/p$ [-]",
        ls=ls,
        color=color,
    )

    make_plot(
        ax.flat[5],
        inertial_number_stack,
        q_p_stack,
        xlabel="$I$ [-]",
        ylabel="$q/p$ [-]",
        ls=ls,
        color=color,
    )

    return fig, ax


def plot_set1_short(
    p_stack,
    q_vm_stack,
    gamma_stack,
    specific_volume_stack,
    inertial_number_stack,
    fig_ax=None,
    **kwargs,
):
    """

    ax_lim = [[xmin,xmax],[ymin,ymax]]

    Args:
        p_stack: _description_
        q_vm_stack: _description_
        gamma_stack: _description_
        dgammadt_stack: _description_
        specific_volume_stack: _description_
        inertial_number_stack: _description_
        fig_ax: _description_. Defaults to None.

    Returns:
        _description_
    """
    if fig_ax is None:
        fig, ax = plt.subplots(
            ncols=3, nrows=1, figsize=(7.5 * 0.9, 4.5 * 0.9), dpi=300
        )
    else:
        fig, ax = fig_ax

    q_p_stack = q_vm_stack / p_stack

    ls = kwargs.get("ls", "-")
    color = kwargs.get("color", None)
    label = kwargs.get("label", None)
    lw = kwargs.get("lw", None)
    make_plot(
        ax.flat[0],
        p_stack,
        q_vm_stack,
        xlabel="$p$ [Pa]",
        ylabel="$q$ [Pa]",
        ls=ls,
        lw=lw,
        color=color,
        label=label,
    )

    make_plot(
        ax.flat[1],
        p_stack,
        specific_volume_stack,
        xlogscale=True,
        ylogscale=True,
        xlabel="$p$ [Pa] (log-scale)",
        ylabel="$v=\\phi^{-1}$ [-] (log-scale)",
        ls=ls,
        color=color,
        lw=lw,
    )

    make_plot(
        ax.flat[2],
        gamma_stack,
        q_p_stack,
        xlabel="$\\gamma$ [-]",
        ylabel="$q/p$ [-]",
        ls=ls,
        color=color,
        lw=lw,
    )

    make_plot(
        ax.flat[3],
        specific_volume_stack,
        q_p_stack,
        ylabel="$q/p$ [-]",
        xlabel="$v=\\phi^{-1}$ [-]",
        ls=ls,
        color=color,
        lw=lw,
    )

    return fig, ax


# from functools import partial
# from typing import Dict, Tuple

# import chex

# from ..utils.math_helpers import (
#     get_hencky_strain_stack,
#     get_pressure_stack,
#     get_q_vm_stack,
#     get_scalar_shear_strain_stack,
#     get_sym_tensor_stack,
#     get_volumetric_strain_stack,
#     phi_to_e_stack,
# )
# from .plot import PlotHelper, make_plots


# def plot_set1(
#     stress_stack: chex.Array,
#     phi_stack: chex.Array,
#     L_stack: chex.Array,
#     plot_helper_args: Dict = None,
#     fig_ax: Tuple = None,
# ) -> Tuple:
#     """Create the plot set 1.

#     Plots the following:

#     q vs p | e-log p | M - e
#     deps_v_dt vs dgamma_dt | log p - phi |  M - phi

#     Args:
#         stress_stack (chex.Array): list of stress tensors
#         phi_stack (chex.Array): list of solid volume fractions
#         L_stack (chex.Array): list of velocity gradients
#         fig_ax (Tuple, optional): Tup. Defaults to None.

#     Returns:
#         Tuple: Fig axis pair
#     """

#     # pass arguments to plot helper from outside
#     if plot_helper_args is None:
#         plot_helper_args = {}

#     _PlotHelper = partial(PlotHelper, **plot_helper_args)

#     # Plot 1: q vs p
#     p_stack = get_pressure_stack(stress_stack)

#     q_stack = get_q_vm_stack(stress_stack)

#     plot1_qp = _PlotHelper(
#         x=p_stack,
#         y=q_stack,
#         xlabel="$p$ [Pa]",
#         ylabel="$q$ [Pa]",
#         # xlogscale=True,
#         # ylogscale=True
#         xlim=[0, p_stack.max() * 1.2],
#         ylim=[0, q_stack.max() * 1.2],
#     )

#     # Plot 2: e-log p
#     e_stack = phi_to_e_stack(phi_stack)

#     plot2_elnp = _PlotHelper(
#         x=p_stack,
#         y=e_stack,
#         xlabel="ln $p$ [-]",
#         ylabel="$e$ [-]",
#         xlim=[0, None],
#         ylim=[0, e_stack.max() * 1.2],
#         xlogscale=True,
#     )

#     # Plot 3: m - e
#     M_stack = q_stack / p_stack

#     plot3_eM = _PlotHelper(
#         x=M_stack,
#         y=e_stack,
#         xlabel="$q/p$ [-]",
#         ylabel="$e$ [-]",
#         xlim=[M_stack.min() * 0.99, M_stack.max() * 1.01],
#         ylim=[0, e_stack.max() * 1.2],
#     )

#     # Plot 4: deps_v_dt vs dgamma_dt
#     deps_dt_stack = get_sym_tensor_stack(L_stack)
#     dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)
#     deps_v_dt_stack = get_volumetric_strain_stack(deps_dt_stack)

#     plot4_deps_v_dt_dgamma_dt = _PlotHelper(
#         x=deps_v_dt_stack,
#         y=dgamma_dt_stack,
#         xlabel="$\dot\\varepsilon_v$ [-]",
#         ylabel="$\dot\\gamma$ [-]",
#         xlim=[deps_v_dt_stack.min() * 0.8, deps_v_dt_stack.max() * 1.2],
#         ylim=[dgamma_dt_stack.min() * 0.8, dgamma_dt_stack.max() * 1.2],
#     )

#     # Plot 5: log p - phi
#     plot5_lnp_phi = _PlotHelper(
#         x=phi_stack,
#         y=p_stack,
#         xlabel="$\phi$ [-]",
#         ylabel="ln $p$ [-]",
#         ylogscale=True,
#         xlim=[phi_stack.min() * 0.99, phi_stack.max() * 1.01],
#         ylim=[p_stack.min() * 0.1, p_stack.max() * 10],  # adjust for logscale
#     )

#     # Plot 6: q/p - phi
#     plot6_Mphi = _PlotHelper(
#         y=M_stack,
#         x=phi_stack,
#         xlabel="$\phi$ [-]",
#         ylabel="$q/p$ [-]",
#         xlim=[phi_stack.min() * 0.99, phi_stack.max() * 1.01],
#         ylim=[M_stack.min() * 0.99, M_stack.max() * 1.01],
#     )

#     fig_ax = make_plots(
#         [
#             plot1_qp,
#             plot2_elnp,
#             plot3_eM,
#             plot4_deps_v_dt_dgamma_dt,
#             plot5_lnp_phi,
#             plot6_Mphi,
#         ],
#         fig_ax=fig_ax,
#     )
#     return fig_ax


# def plot_set2(
#     stress_stack: chex.Array,
#     L_stack: chex.Array,
#     F_stack: chex.Array,
#     plot_helper_args: Dict = None,
#     fig_ax: Tuple = None,
# ) -> Tuple:
#     """Create the plot set 1.

#     Plots included:

#     q vs gamma | p vs gamma | M vs gamma
#     q vs dot gamma | p vs dot gamma | M vs dot gamma

#     Args:
#         stress_stack (chex.Array): list of stress tensors
#         L_stack (chex.Array): list of velocity gradients
#         F_stack (chex.Array): list of deformation gradients
#         fig_ax (Tuple, optional): fig axis pair. Defaults to None.

#     Returns:
#         Tuple: Update fig axes pair
#     """

#     # pass arguments to plot helper from outside
#     if plot_helper_args is None:
#         plot_helper_args = {}

#     _PlotHelper = partial(PlotHelper, **plot_helper_args)

#     # Plot 1: q - gamma
#     eps_stack, *_ = get_hencky_strain_stack(F_stack)

#     gamma_stack = get_scalar_shear_strain_stack(eps_stack)

#     q_stack = get_q_vm_stack(stress_stack)

#     plot1_q_gamma = _PlotHelper(
#         x=gamma_stack,
#         y=q_stack,
#         xlabel="$\gamma$ [-]",
#         ylabel="$q$ [Pa]",
#         xlim=[0, gamma_stack.max() * 1.2],
#         ylim=[0, q_stack.max() * 1.2],
#     )

#     # Plot 2: p vs gamma
#     p_stack = get_pressure_stack(stress_stack)

#     plot2_p_gamma = _PlotHelper(
#         x=gamma_stack,
#         y=p_stack,
#         xlabel="$\gamma$ [-]",
#         ylabel="$p$ [Pa]",
#         xlim=[0, gamma_stack.max() * 1.2],
#         ylim=[0, p_stack.max() * 1.2],
#     )

#     # Plot 3: M vs gamma
#     p_stack = get_pressure_stack(stress_stack)
#     M_stack = q_stack / p_stack

#     plot2_M_gamma = _PlotHelper(
#         x=gamma_stack,
#         y=M_stack,
#         xlabel="$\gamma$ [-]",
#         ylabel="$q/p$ [-]",
#         xlim=[0, gamma_stack.max() * 1.2],
#         ylim=[M_stack.min() * 0.99, M_stack.max() * 1.01],
#     )

#     # Plot 4: q vs dot gamma
#     q_stack = get_q_vm_stack(stress_stack)

#     deps_dt_stack = get_sym_tensor_stack(L_stack)

#     dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)

#     plot4_q_dgamma_dt = _PlotHelper(
#         x=dgamma_dt_stack,
#         y=q_stack,
#         xlabel="$\dot\gamma$ [-]",
#         ylabel="$q$ [Pa]",
#         xlim=[0, dgamma_dt_stack.max() * 1.2],
#         ylim=[0, q_stack.max() * 1.2],
#     )

#     # Plot 5: p vs dot gamma
#     plot5_p_dgamma_dt = _PlotHelper(
#         x=dgamma_dt_stack,
#         y=p_stack,
#         xlabel="$\dot\gamma$ [-]",
#         ylabel="$p$ [Pa]",
#         xlim=[0, dgamma_dt_stack.max() * 1.2],
#         ylim=[0, p_stack.max() * 1.2],
#     )

#     # Plot 6: M vs dot gamma

#     plot6_M_dgamma_dt = _PlotHelper(
#         x=dgamma_dt_stack,
#         y=M_stack,
#         xlabel="$\dot\gamma$ [-]",
#         ylabel="$q/p$ [-]",
#         xlim=[0, dgamma_dt_stack.max() * 1.2],
#         ylim=[M_stack.min() * 0.99, M_stack.max() * 1.01],
#     )

#     fig_ax = make_plots(
#         [
#             plot1_q_gamma,
#             plot2_p_gamma,
#             plot2_M_gamma,
#             plot4_q_dgamma_dt,
#             plot5_p_dgamma_dt,
#             plot6_M_dgamma_dt,
#         ],
#         fig_ax=fig_ax,
#     )

#     return fig_ax


# def plot_set3(
#     stress_stack: chex.Array,
#     phi_stack: chex.Array,
#     L_stack: chex.Array,
#     F_stack: chex.Array,
#     t_stack: chex.Array,
#     plot_helper_args: Dict = None,
#     fig_ax: Tuple = None,
# ):
#     """Create plot set 3:

#     Plots include:
#     q - t | p - t | M - t |
#     phi - t | gamma -t | dgamma_dt - t

#     Args:
#         stress_stack (chex.Array): list of stress tensors
#         phi_stack (chex.Array): list of solid volume fractions
#         L_stack (chex.Array): list of velocity gradients
#         F_stack (chex.Array): list of deformation gradients
#         t_stack (chex.Array): time stack

#     Returns:
#         Typle: Updated fix axes pair
#     """

#     # pass arguments to plot helper from outside
#     if plot_helper_args is None:
#         plot_helper_args = {}

#     _PlotHelper = partial(PlotHelper, **plot_helper_args)

#     # Plot 1: q - t

#     q_stack = get_q_vm_stack(stress_stack)

#     plot1_q_t = _PlotHelper(
#         x=t_stack,
#         y=q_stack,
#         xlabel="$t$ [s]",
#         ylabel="$q$ [Pa]",
#         xlogscale=True,
#         ylogscale=True,
#         # ylim=[0, q_stack.max()*1.2],
#     )

#     # Plot 2: p - t

#     p_stack = get_pressure_stack(stress_stack)

#     plot2_p_t = _PlotHelper(
#         x=t_stack,
#         y=p_stack,
#         xlabel="$t$ [s]",
#         ylabel="$p$ [Pa]",
#         ylim=[0, p_stack.max() * 1.2],
#         xlogscale=True,
#         ylogscale=True,
#     )

#     # Plot 3: M - t
#     M_stack = q_stack / p_stack
#     plot3_M_t = _PlotHelper(
#         x=t_stack,
#         y=M_stack,
#         xlabel="$t$ [s]",
#         ylabel="$q/p$ [-]",
#         # ylim=[M_stack.min()*0.99, M_stack.max()*1.01],
#         xlogscale=True,
#         ylogscale=True,
#     )

#     # Plot 4: phi - t

#     plot4_phi_t = _PlotHelper(
#         x=t_stack,
#         y=phi_stack,
#         xlabel="$t$ [s]",
#         ylabel="$\phi$ [-]",
#         ylim=[phi_stack.min() * 0.99, phi_stack.max() * 1.01],
#         xlogscale=True,
#         # ylogscale=True
#     )

#     # Plot 5: gamma - t
#     eps_stack, *_ = get_hencky_strain_stack(F_stack)
#     gamma_stack = get_scalar_shear_strain_stack(eps_stack)

#     plot5_gamma_t = _PlotHelper(
#         x=t_stack,
#         y=gamma_stack,
#         xlabel="$t$ [s]",
#         ylabel="$\gamma$ [-]",
#         ylim=[gamma_stack.min() * 0.9, gamma_stack.max() * 1.1],
#         xlogscale=True,
#     )

#     # Plot 6: dot gamma - t

#     deps_dt_stack = get_sym_tensor_stack(L_stack)

#     dgamma_dt_stack = get_scalar_shear_strain_stack(deps_dt_stack)

#     plot6_dgamma_dt_t = _PlotHelper(
#         x=t_stack,
#         y=dgamma_dt_stack,
#         xlabel="$t$ [s]",
#         ylabel="$\dot\gamma$ [-]",
#         ylim=[dgamma_dt_stack.min() * 0.9, dgamma_dt_stack.max() * 1.1],
#         xlogscale=True,
#     )

#     fig_ax = make_plots(
#         [
#             plot1_q_t,
#             plot2_p_t,
#             plot3_M_t,
#             plot4_phi_t,
#             plot5_gamma_t,
#             plot6_dgamma_dt_t,
#         ],
#         fig_ax=fig_ax,
#     )

#     return fig_ax
