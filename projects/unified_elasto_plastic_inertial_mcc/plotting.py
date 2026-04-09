import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
from typing import Tuple
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from UEPI_MCC import get_iss_M, get_unified_M, get_iss_v, get_v_csl, get_v_ncl

###########################################################################
# Configure directories
###########################################################################

ROOT_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"
FIGURE_DIR = ROOT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

###########################################################################
# Matplotlib Configuration
###########################################################################

plt.rcParams.update(
    {
        "figure.figsize": [3.5, 2.625],
        "font.size": 12,
        "axes.labelsize": 11,
        "xtick.direction": "in",
        "xtick.major.size": 4,
        "xtick.major.width": 0.5,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.5,
        "xtick.minor.visible": True,
        "xtick.top": True,
        "ytick.direction": "in",
        "ytick.major.size": 3,
        "ytick.major.width": 0.5,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.5,
        "ytick.minor.visible": True,
        "ytick.right": True,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "legend.loc": "upper right",
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.handlelength": 1.5,
        "legend.fontsize": 8,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }
)

###########################################################################
# Plotting colors
###########################################################################

COLORS = {
    "ncl": "black",
    "csl": "tab:red",
    "sl": "tab:blue",
    "ys": "tab:blue",
}

LABELS = {
    "pressure": "pressure $p$ [kPa]",
    "deviatoric_stress": "deviatoric stress $q$ [kPa]",
    "inertial_number": "inertial number $I$ [--]",
    "plastic_inertial_number": "plastic inertial number $I^p$ [--]",
    "solid_volume_fraction": "solid volume fraction $\\phi=v^{-1}$ [--]",
    "stress_ratio": "stress_ratio $M=\\sqrt{3}\\mu$ [--]",
    "specific_volume": "specific volume $v=\\phi^{-1}$ [--]",
    "shear_strain_perc": "deviatoric strain $\\varepsilon_q$ [\%]",
    "volumetric_strain": "volumetric strain $\\varepsilon_v$ [\%]",
    "axial_strain_perc": "axial strain $\\varepsilon_a$ [\%]",
}

FIG_SIZES = {
    "full_page_2": (7.0 * 0.95, 5.25 * 0.95),
    "full_page_2_extended": (7.0 * 0.95, 5.25),
    "single_col": (3.5, 2.625),
}


###########################################################################
# Helper functions for plotting
###########################################################################


def annotate_axes(ax, label: str) -> None:
    """Add annotation to top-left corner of axes."""
    ax.annotate(
        label,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(+0.25, -0.25),
        textcoords="offset fontsize",
        fontsize=12,
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.8", alpha=0.5, edgecolor="none", pad=3.0),
    )


def annotate_point(
    ax,
    label: str,
    xy: Tuple[float, float],
    arrow: bool = False,
    xytext: Tuple[float, float] = None,
    xycoords: str = "data",
    angle: float = 0,
    color: str = "black",
    fontsize: int = 11,
) -> None:
    """
    Add annotation to plot with optional arrow pointer.
    """
    arrowprops = (
        dict(
            facecolor="black",
            edgecolor="black",
            arrowstyle="->",
            shrinkA=5,
            shrinkB=5,
            lw=1.5,
        )
        if arrow
        else None
    )

    ax.annotate(
        label,
        xy=xy,
        xytext=xytext,
        xycoords=xycoords,
        arrowprops=arrowprops,
        rotation=angle,
        rotation_mode="anchor",
        fontsize=fontsize,
        color=color,
        ha="center",
        va="center",
        annotation_clip=False,
    )


def plot_triangle(
    ax, p1: Tuple[float, float], p2: Tuple[float, float], upside: bool = True
) -> Tuple[float, float]:
    """
    Plot triangle slope indicators on axes.
    """
    x1, x2 = p1[0], p2[0]
    y1, y2 = p1[1], p2[1]

    if upside:
        ax.plot([x1, x2, x2], [y1, y1, y2], ls="-", lw=1.0, color="black")
        return (x2, y1)

    ax.plot([x1, x1, x2], [y1, y2, y2], ls="-", lw=1.0, color="black")
    return (x1, y1)


def plot_orthogonality_marker(ax, origin, normal, tangent, size=5.0):
    """
    Plots a small right-angle marker to indicate orthogonality.
    """

    n = np.array(normal)
    n = n / np.linalg.norm(n)

    t = np.array(tangent)
    t = t / np.linalg.norm(t)

    p0 = np.array(origin)
    p1 = p0 + size * t
    p2 = p0 + size * t + size * n
    p3 = p0 + size * n

    ax.plot([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], "k-", lw=0.5)


def format_sci_notation(val):
    """Converts a number to a LaTeX formatted scientific notation string."""
    if val == 0:
        return "0"

    s = "{:.2e}".format(val)
    base, exponent = s.split("e")

    exponent = int(exponent)
    return rf"{base} \times 10^{{{exponent}}}"


def sort_legend_handles(axes_list, order_list=None):
    """
    Sorts legend handles form a list of axes.
    """
    handles, labels = [], []
    seen = set()

    # Collect unique handles
    for ax in axes_list:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)

    if not order_list:
        order_list = ["CSL", "NCL", "ISS", "SL", "YS", "I-YS"]

    def sort_key(item):
        lbl = item[1]
        if any(x in lbl for x in order_list):
            return 1
        return 0

    # Sort
    zipped = sorted(zip(handles, labels), key=sort_key)
    if not zipped:
        return [], []
    return zip(*zipped)


###########################################################################
# Triaxial dashboard class for numerical results
###########################################################################


class TriaxialDashboard:
    """
    A specialized plotter for Triaxial Compression results.
    Layout:
      (top-left) p, q       (top-right) ln p, ln v
      (bot-left) q, eps_q      (bot-right) eps_v, eqs_q or (p,eps_q)
    """

    def __init__(self, is_undrained: bool, figsize_key="full_page_2_extended"):
        self.is_undrained = is_undrained

        self.fig = plt.figure(figsize=FIG_SIZES[figsize_key])
        gs = gridspec.GridSpec(nrows=2, ncols=2)

        self.ax_pq = self.fig.add_subplot(gs[0, 0])
        self.ax_pq.set_aspect("equal", adjustable="box", anchor="SW")

        self.ax_vp = self.fig.add_subplot(gs[0, 1])
        self.ax_qe = self.fig.add_subplot(gs[1, 0])
        self.ax_aux = self.fig.add_subplot(gs[1, 1])

        self._shared_reference_plotted = False

        self._configure_axes()

        self.traj_handles = []
        self.traj_labels = []

    def _configure_axes(self):

        self.ax_pq.set_xlabel(LABELS["pressure"])
        self.ax_pq.set_ylabel(LABELS["deviatoric_stress"])

        self.ax_vp.set_xlabel(LABELS["pressure"])
        self.ax_vp.set_xscale("log")
        self.ax_vp.xaxis.set_major_formatter(ScalarFormatter())

        self.ax_vp.set_ylabel(LABELS["specific_volume"])
        self.ax_vp.yaxis.set_label_coords(-0.15, 0.42)
        self.ax_vp.set_yscale("log")
        self.ax_vp.yaxis.set_major_formatter(ScalarFormatter())

        self.ax_qe.set_xlabel(LABELS["shear_strain_perc"])
        self.ax_qe.set_ylabel(LABELS["deviatoric_stress"])

        self.ax_aux.set_xlabel(LABELS["shear_strain_perc"])
        if self.is_undrained:
            self.ax_aux.set_ylabel(LABELS["pressure"])
        else:
            self.ax_aux.set_ylabel(LABELS["volumetric_strain"])

        if self.is_undrained:

            annotate_axes(self.ax_pq, "(a) TRX CU")
            annotate_axes(self.ax_vp, "(b) TRX CU")
            annotate_axes(self.ax_qe, "(c) TRX CU")
            annotate_axes(self.ax_aux, "(d) TRX CU")
        else:

            annotate_axes(self.ax_pq, "(e) TRX CD")
            annotate_axes(self.ax_vp, "(f) TRX CD")
            annotate_axes(self.ax_qe, "(g) TRX CD")
            annotate_axes(self.ax_aux, "(h) TRX CD")

    def configure_limits(
        self,
        p_lim=None,
        p_log_lim=None,
        q_lim=None,
        v_lim=None,
        eps_lim=None,
        v_ticks=None,
        p_log_ticks=None,
        eps_v_lim = None,
        eps_v_ticks = None 
    ):
        """
        Sets axis limits across all shared subplots.

        """
        if p_lim is not None:
            self.ax_pq.set_xlim(p_lim)
            self.ax_pq.set_xmargin(0)
            self.ax_pq.set_ymargin(1.5)

        if p_log_lim is not None:
            self.ax_vp.set_xlim(p_log_lim)
            if self.is_undrained:
                self.ax_aux.set_ylim(p_log_lim)
                if p_log_ticks is not None:
                    self.ax_aux.set_yticks(p_log_ticks)
                    self.ax_aux.set_yticklabels(p_log_ticks)

        elif p_lim is not None and p_lim[0] > 0:

            # Fallback: use linear p limits if they are safe for log scale
            self.ax_vp.set_xlim(p_lim)

        if q_lim is not None:
            self.ax_pq.set_ylim(q_lim)
            self.ax_qe.set_ylim(q_lim)

        # specific Volume (with tick fix for small log ranges)
        if v_lim is not None:
            self.ax_vp.set_ylim(v_lim)
            self.ax_vp.yaxis.set_major_formatter(ScalarFormatter())
            self.ax_vp.yaxis.set_minor_formatter(ScalarFormatter())

            self.ax_vp.minorticks_off()

            if v_ticks is not None:
                self.ax_vp.set_yticks(v_ticks)
                self.ax_vp.set_yticklabels([f"{v:.2f}" for v in v_ticks])

            if p_log_ticks is not None:
                self.ax_vp.set_xticks(p_log_ticks)
                self.ax_vp.set_xticklabels(p_log_ticks)

        # volumetric strain if undrained
        if not self.is_undrained:
            self.ax_aux.set_ylim(eps_v_lim)

        # axial Strain
        if eps_lim is not None:
            self.ax_qe.set_xlim(eps_lim)
            self.ax_aux.set_xlim(eps_lim)

    def plot_trajectory(
        self, p, q, v, eps_s, eps_v=None, I_p=0.0, label=0.0, color="k", zorder=0
    ):
        """
        Plots a single simulation trajectory across all 4 subplots.
        """

        # plot lines
        self.ax_pq.plot(p, q, color=color, label=label, zorder=zorder, lw=2)
        self.ax_vp.plot(p, v, color=color, zorder=zorder, lw=2)
        self.ax_qe.plot(eps_s, q, color=color, zorder=zorder, lw=2)

        aux_y = p if self.is_undrained else eps_v

        if aux_y is not None:
            self.ax_aux.plot(eps_s, aux_y, color=color, zorder=zorder, lw=2)

        # plot markers at start and end points
        marker_style_start = dict(
            marker="o",
            markersize=9,
            markeredgecolor="k",
            markeredgewidth=0.5,
            color=color,
            zorder=10,
        )
        marker_style_end = dict(
            marker="*",
            markersize=12,
            markeredgecolor="k",
            markeredgewidth=0.5,
            color=color,
            zorder=10,
        )

        # Start markeer
        self.ax_pq.plot(p[0], q[0], **marker_style_start)
        self.ax_vp.plot(p[0], v[0], **marker_style_start)
        self.ax_qe.plot(eps_s[0], q[0], **marker_style_start)
        if aux_y is not None:
            self.ax_aux.plot(eps_s[0], aux_y[0], **marker_style_start)

        formatted_ip = format_sci_notation(I_p)
        label_text = rf"$I^p: 0 \to {formatted_ip}$"

        self.ax_pq.plot(p[-1], q[-1], **marker_style_end)
        self.ax_vp.plot(p[-1], v[-1], **marker_style_end)
        (line,) = self.ax_qe.plot(eps_s[-1], q[-1], **marker_style_end)
        if aux_y is not None:
            self.ax_aux.plot(eps_s[-1], aux_y[-1], **marker_style_end)

        self.traj_handles.append(line)
        self.traj_labels.append(label_text)

    def add_trajectory_reference(
        self, sim_config, p, q, v, I_p, color, i_ys_label=None, i_ss_label=None
    ):
        """Add reference lines for each trajectory"""

        p_max = self.ax_pq.get_xlim()[1]
        p_ys = jnp.linspace(0.1 / 1000, p_max * 1000, 50_000)
        p_sl = jnp.array([1000, sim_config.confine * sim_config.OCR])

        # iss line (p,q)
        pq_iss = get_iss_M(
            I=I_p, I_M=sim_config.I_M, M_csl=sim_config.M_csl, M_inf=sim_config.M_inf
        )

        self.ax_pq.plot(
            [0, p_max],
            [0, pq_iss * p_max],
            color=color,
            linestyle="--",
            zorder=-1,
            lw=1.0,

        )

        # yield surface (p,q)
        p_s = p[-1] * 1000

        ys = get_unified_M(
            xi=p_s / p_ys,
            I=I_p,
            M_csl=sim_config.M_csl,
            M_inf=sim_config.M_inf,
            I_M=sim_config.I_M,
        )

        q_ys = ys * p_ys

        self.ax_pq.plot(
            p_ys / 1000,
            q_ys / 1000,
            color=color,
            linestyle=":",
            label=i_ys_label,
            zorder=-1,
            lw=1.0,
        )

        # ISS in bi-logarithmic (p,v)
        v_csl = get_v_csl(
            p_s=jnp.array([1.0, p_max * 1000]),
            p_ref=1000,
            gamma=sim_config.gamma,
            lam=sim_config.lam,
        )
        v_iss = get_iss_v(v_csl, I_p, sim_config.I_v)

        self.ax_vp.plot(
            jnp.array([1e-3, p_max]),
            v_iss,
            color=color,
            linestyle="--",
            zorder=-1,
            lw=1.0,
            label=i_ss_label,
        )

    def add_reference(
        self,
        sim_config,
    ):
        """Add reference lines shared by all trajectories"""

        p_max = self.ax_pq.get_xlim()[1]
        p_ys = jnp.linspace(1 / 1000, p_max * 1000, 5_000)

        # CSL line
        self.ax_pq.plot(
            [0, p_max],
            [0, sim_config.M_csl * p_max],
            color=COLORS["csl"],
            linestyle="-",
            label="CSL (ref.)",
            zorder=-1,
            lw=0.5
        )

        # initial yield surface (p,q)
        p_s = sim_config.OCR * sim_config.confine / 2.0

        qp = get_unified_M(
            xi=p_s / p_ys,
            I=0.0,
            M_csl=sim_config.M_csl,
            M_inf=sim_config.M_inf,
            I_M=sim_config.I_M,
        )

        q_ys = qp * p_ys

        self.ax_pq.plot(
            p_ys / 1000,
            q_ys / 1000,
            color=COLORS["sl"],
            linestyle="-",
            label="YS (init.)",
            zorder=0,
            lw=0.5,
            alpha=0.75,
        )

        # NCL line in bi-logarithmic (p,v)
        v_ncl = get_v_ncl(
            p_c=jnp.array([1, p_max * 1000]),
            p_ref=1000.0,
            N=sim_config.N,
            lam=sim_config.lam,
        )

        self.ax_vp.plot(
            jnp.array([1e-3, p_max]),
            v_ncl,
            color=COLORS["ncl"],
            linestyle="-",
            lw=0.5,
            alpha=0.75,
            zorder=-1,
            label="NCL (ref.)",
        )

        # CSL line in bi-logarithmic (p,v)
        v_csl = get_v_csl(
            p_s=jnp.array([1, p_max * 1000]),
            p_ref=1000.0,
            gamma=sim_config.gamma,
            lam=sim_config.lam,
        )

        self.ax_vp.plot(
            jnp.array([1e-3, p_max]),
            v_csl,
            color=COLORS["csl"],
            linestyle="-",
            zorder=-1,
            lw=0.5,
        )

    def finalize(
        self,
        filename: str,
        ncol: int = 3,
        hide_legend: bool = False,
        bottom_space: float = 0.28,
    ):
        from matplotlib.legend_handler import HandlerTuple

        self.fig.subplots_adjust(
            bottom=bottom_space,
            hspace=0.30,
            right=0.98,
            left=0.1,
            wspace=0.25,
            top=0.95,
        )

        if self.traj_handles:
            numeric_legend = self.ax_qe.legend(
                self.traj_handles,
                self.traj_labels,
                loc="lower right",
                fontsize=9,
                frameon=True,
                handlelength=1.0,
                borderpad=0.5,
                labelspacing=0.2,
                handletextpad=0.4,
                borderaxespad=0.2,
            )
            self.ax_qe.add_artist(numeric_legend)

        if not hide_legend:
            axes = [self.ax_pq, self.ax_vp, self.ax_qe, self.ax_aux]
            order_list = [
                r"$\dot{\varepsilon}_{a,1}=$",
                "YS",
                "I-YS",
                "CSL",
                "NCL",
            ]
            handles, labels = sort_legend_handles(axes, order_list=order_list)

            if handles:
                self.fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=ncol,
                    bbox_to_anchor=(0.5, 0.05),
                    frameon=False,
                    fontsize=10,
                    columnspacing=0.5,
                    handlelength=1.0,
                )

        print(f"Saving to {filename}")
        self.fig.savefig(f"{filename}.png", dpi=300)
        self.fig.savefig(f"{filename}.pdf", dpi=300)

        plt.close(self.fig)
