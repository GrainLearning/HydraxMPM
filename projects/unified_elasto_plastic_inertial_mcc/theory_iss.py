import numpy as np
import matplotlib.pyplot as plt
import sys


import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from UEPI_MCC import get_M_I, get_v_I, get_iss_M, get_iss_v

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

M_csl = 1.0  # critical state bulk friction [-]
M_inf = 2.0  # dynamic bulk friction at high inertial numbers [-]
I_M = 0.279  # charateristic inertial number for friction [-]
I_v = 0.1  # charateristic inertial number for dilation [-]
v_csl = 2  # critical state specific volume [-]

###########################################################################
# Inertial steady state rheology
###########################################################################

I_stack = np.logspace(-4, 4, 5000)

M_I_stack = get_M_I(I_stack, I_M, M_csl, M_inf)


v_I_stack = get_v_I(I_stack, I_v)

v_stack = get_iss_v(v_csl, I_stack, I_v)
M_stack = get_iss_M (I_stack, I_M, M_csl, M_inf)

###########################################################################
# Setup, plot, save (I, M)
###########################################################################

fig = plt.figure(figsize=FIG_SIZES["single_col"])
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)


ax.plot(
    [0, 1000_000],
    [M_csl, M_csl],
    color=COLORS["csl"],
    linestyle="-",
    label="CSL",
)

ax.plot(
    I_stack,
    M_stack,
    color=COLORS["csl"],
    linestyle="--",
    label="ISS",
)


annotate_point(
    ax,
    "$M \\to M_\\textsc{csl}$",
    (0.0007, M_csl * 1.1),
    arrow=False,
    xytext=(0.0007, M_csl * 1.1),
)
annotate_point(
    ax,
    "$M \\to M_\\infty$",
    (120, M_inf * 0.9),
    arrow=False,
    xytext=(120, M_inf * 0.95),
)

# single logarithmic x-axis
ax.set_xscale("log")

ax.set_xticks([], minor=True)
ax.set_xticks([], minor=False)

ax.set_yticks([], minor=True)
ax.set_yticks([], minor=False)

# Set axis limits
ax.set_ylim(M_csl * 0.9, M_inf * 1.15)

# Set axis labels
ax.set_xlim(0.00006, 1000)
ax.set_xlabel(LABELS["plastic_inertial_number"])
ax.set_ylabel(LABELS["stress_ratio"])
ax.yaxis.set_label_coords(-0.01, 0.46)

# Annotate axes
annotate_axes(ax, "(b)")

# Setup legend
handles, labels = sort_legend_handles([ax])
ax.legend(handles, labels, loc="upper right", ncols=2)

# save figure
fig.savefig(FIGURE_DIR / "theory_M_I.pdf", bbox_inches="tight")
print("Saved figure to:", FIGURE_DIR / "theory_M_I.pdf")


###########################################################################
# Setup, plot, save (I,v)
###########################################################################

fig = plt.figure(figsize=FIG_SIZES["single_col"])
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)

ax.plot(
    [0, 1000],
    [v_csl, v_csl],
    color=COLORS["csl"],
    linestyle="-",
    label="CSL",
)

ax.plot(
    I_stack,
    v_stack,
    color=COLORS["csl"],
    linestyle="--",
    label="ISS",
)


ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([], minor=True)
ax.set_xticks([], minor=False)

ax.set_yticks([], minor=True)
ax.set_yticks([], minor=False)

ax.set_xlim(0.0001, 500)
ax.set_ylim(v_csl * 0.5, v_csl * 1e10)

ax.set_xlabel(LABELS["plastic_inertial_number"])
ax.set_ylabel(LABELS["specific_volume"])
ax.yaxis.set_label_coords(-0.01, 0.47)


annotate_point(
    ax,
    "$v \\to v_\\textsc{csl}$",
    (0.0001, v_csl * 5.0),
    arrow=False,
    xytext=(0.0007, v_csl * 5.0),
)
annotate_point(
    ax, "$v \\to \infty $", (0.38, v_csl * 1e9), arrow=False, xytext=(0.38, v_csl * 1e9)
)

annotate_axes(ax, "(a)")
handles, labels = sort_legend_handles([ax])
ax.legend(handles, labels, loc="upper right", ncols=1)
fig.savefig(FIGURE_DIR / "theory_v_I.pdf", bbox_inches="tight")
