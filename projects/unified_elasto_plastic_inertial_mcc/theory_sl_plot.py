import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from UEPI_MCC import get_xi, convert_gamma_N, get_v_csl, get_mcc_sl, get_v_ncl

from plotting import (
    COLORS,
    FIG_SIZES,
    LABELS,
    FIGURE_DIR,
    annotate_axes,
    sort_legend_handles,
    annotate_point,
    plot_triangle,
)

###########################################################################
# Material parameters
###########################################################################

lam = 0.10  # slope of csl [-]
kap = lam / 5  # slope of swelling line [-]
p_ref = 1_000  # 1kPa reference pressure [Pa]
gamma = 4.8  # intercept of CSL at p_ref [-]
N = convert_gamma_N(gamma=gamma, lam=lam, kap=kap)


###########################################################################
# Compute State variables & state parameter
###########################################################################

p_s = 1_750  # shift pressure [Pa]
p_c = 2 * p_s  # normal consolidation pressure [Pa]

p_stack = np.arange(p_ref, 100_000)  # for plotting CSL, NCL

p_sl_stack = np.arange(p_ref, p_c, 1)  # for plotting SL


###########################################################################
# Compute product functions
###########################################################################
v_csl = get_v_csl(p_s, p_ref, gamma, lam)
v_ncl = get_v_ncl(p_c, p_ref, N, lam)


# Reference functions for plotting
v_csl_ref_stack = get_v_csl(p_stack, p_ref, gamma, lam)
v_ncl_ref_stack = get_v_ncl(p_stack, p_ref,  N, lam,)

# Swelling line state function

xi_stack = get_xi(p_s, p_sl_stack)
v_stack = get_mcc_sl(xi_stack, p_s, p_ref, gamma, lam, kap)

###########################################################################
# Setup & plot figures
###########################################################################

fig = plt.figure(figsize=FIG_SIZES["single_col"])
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)

ax.plot(
    p_sl_stack / 1000,
    v_stack,
    color=COLORS["sl"],
    linestyle="-",
    label="SL",
)

# Add reference lines
ax.plot(
    p_stack / 1000,
    v_csl_ref_stack,
    color=COLORS["csl"],
    linestyle="-",
    label="CSL",
)

ax.plot(
    p_stack / 1000,
    v_ncl_ref_stack,
    color=COLORS["ncl"],
    linestyle="-",
    label="NCL",
)

# annotate (p_c, v_ncl) point


ax.plot(
    [p_c / 1000],
    [v_ncl],
    marker="o",
    markersize=4,
    color="black",
    markeredgecolor="black",
    zorder=4,
)
annotate_point(
    ax, "$(p_\mathrm{c},v_\\textsc{ncl})$", (p_c / 1000 - 0.1, v_ncl + 0.0999)
)

# annotate (p_ref, N) point
ax.plot(
    [p_ref / 1000],
    [N],
    marker="o",
    markersize=4,
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
)
annotate_point(ax, "$(p_\mathrm{ref},N)$", (p_ref / 1000 + 0.205, N + 0.0070))

# annotate (p_s, Gamma) point
ax.plot(
    [p_ref / 1000],
    [gamma],
    marker="o",
    markersize=4,
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
)
annotate_point(ax, "$(p_\mathrm{ref},\Gamma)$", (p_ref / 1000 + 0.205, gamma + 0.0070))

# Point (p_s, v_s)
ax.plot(
    [p_s / 1000],
    [v_csl],
    marker="o",
    markersize=4,
    color="black",
    markeredgecolor="black",
    zorder=4,
)
annotate_point(
    ax, "$(p_\mathrm{s},v_\\textsc{csl})$", (p_s / 1000 - 0.3, v_csl - 0.045)
)

# Slope of NCL (lambda)
xl, yl = plot_triangle(
    ax,
    (p_stack[400] / 1000, v_ncl_ref_stack[400]),
    (p_stack[600] / 1000, v_ncl_ref_stack[600]),
)
annotate_point(ax, "$\lambda$", (xl + 0.05, yl - 0.03))

# Slope of CSL (lambda)
xl, yl = plot_triangle(
    ax,
    (p_stack[400] / 1000, v_csl_ref_stack[400]),
    (p_stack[600] / 1000, v_csl_ref_stack[600]),
)
annotate_point(ax, "$\lambda$", (xl + 0.05, yl - 0.03))

# Slope of CSL (lambda)
xl, yl = plot_triangle(
    ax,
    (p_sl_stack[400] / 1000, v_stack[400]),
    (p_sl_stack[600] / 1000, v_stack[600]),
    upside=False,
)
annotate_point(ax, "$\kappa$", (xl - 0.06, yl - 0.015))


# spacing ratio
ax.plot(
    [p_c / 1000, p_c / 1000],
    [ax.get_ylim()[0], v_ncl],
    color="black",
    linestyle="dashed",
    lw=1.0,
    zorder=1,
)

ax.plot(
    [p_s / 1000, p_s / 1000],
    [ax.get_ylim()[0], v_csl],
    color="black",
    linestyle="dashed",
    lw=1.0,
    zorder=1,
)


ax.annotate(
    "",
    xy=(p_s / 1000, v_ncl * 0.98),
    xytext=(p_c / 1000, v_ncl * 0.98),
    arrowprops=dict(
        arrowstyle="->",
        color="black",
        lw=1.0,
    ),
    annotation_clip=False,
)

annotate_point(ax, "$p_\\mathrm{c}$/2", (((p_c + p_s)*1.075) / 2 / 1000, v_ncl * 0.991))


# ###########################################################################
# # Configure plots
# ###########################################################################

# Bi-logarithmic scale
ax.set_xscale("log")
ax.set_yscale("log")

ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())


# Hide ticks
ax.set_xticks([], minor=False)
ax.set_xticks([], minor=True)
ax.set_yticks([], minor=False)
ax.set_yticks([], minor=True)

# Set axis limits
ax.set_xlim(0.83, 4)
ax.set_ylim(gamma * 0.91, gamma * 1.07)

# Set axis labels
ax.set_xlabel(LABELS["pressure"])
ax.set_ylabel(LABELS["specific_volume"])

# Annotate axes
annotate_axes(ax, "(a)")

# Setup legend
handles, labels = sort_legend_handles([ax])
ax.legend(handles, labels, loc="upper right")

# save figure
fig.savefig(FIGURE_DIR / "theory_swelling_line.pdf", bbox_inches="tight")
print("Saved figure to:", FIGURE_DIR / "theory_swelling_line.pdf")
