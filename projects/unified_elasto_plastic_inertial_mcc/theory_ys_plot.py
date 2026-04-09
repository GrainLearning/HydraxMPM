import numpy as np
import matplotlib.pyplot as plt
import sys


import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


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

from UEPI_MCC import get_xi, get_M_xi, get_mcc_M

###########################################################################
# Material parameters
###########################################################################

M_csl = 1.0  # critical state friction coefficient [-]
xi_0 = 1.0 / 2.0  # spacing parameter [-]


###########################################################################
# Compute State variables & state parameter
###########################################################################

p_s = 100_000  # shift pressure [Pa]
p_c = 2 * p_s  # normal consolidation pressure [Pa]

p_stack = np.linspace(1e-12, p_c * 10, 50_000)

xi_stack = get_xi(p_s, p_stack)


###########################################################################
# Compute product functions
###########################################################################


M_stack = get_mcc_M(xi=xi_stack, M_csl=M_csl)

q_M_csl_stack = M_csl * p_stack
q_M_stack = M_stack * p_stack

###########################################################################
# Setup & plot figures
###########################################################################

fig = plt.figure(figsize=FIG_SIZES["single_col"])
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal", adjustable="datalim")


ax.plot(
    p_stack / 1000,
    q_M_stack / 1000,
    color=COLORS["ys"],
    linestyle="-",
    label="YS",
)

# Add reference lines
ax.plot(
    p_stack / 1000,
    q_M_csl_stack / 1000,
    color=COLORS["csl"],
    linestyle="-",
    label="CSL",
)

ax.plot(
    [0, 1000],
    [0.0, 0.0],
    color=COLORS["ncl"],
    linestyle="-",
    label="NCL",
)

start_end = (3000, 3500)
xl, yl = plot_triangle(
    ax,
    (p_stack[start_end[0]] / 1000, q_M_csl_stack[start_end[0]] / 1000),
    (p_stack[start_end[1]] / 1000, q_M_csl_stack[start_end[1]] / 1000),
)
annotate_point(ax, "$M_\\textsc{csl}$", (xl + 20.0, yl + 0.95))


# Add annotations
ax.plot(
    [(p_s) / 1000],
    [(p_s * M_csl) / 1000],
    marker="o",
    markersize=4,
    color="black",
    markeredgecolor="black",
    zorder=4,
)

annotate_point(
    ax, "$(p_\\mathrm{s}, q_\\mathrm{s})$", (0.80 * (p_s) / 1000, 1.15 * (((p_s) * M_csl) / 1000))
)

ax.plot(
    [(p_c) / 1000],
    [0.0],
    marker="o",
    markersize=4,
    color="black",
    markeredgecolor="black",
    zorder=4,
)

annotate_point(ax, "$(p_\\mathrm{c}, 0)$", (1.15 * p_c / 1000, 12.5))


y_pad = 10.0


ax.annotate(
    "",
    xy=(p_s / 1000, 0.0 + y_pad),
    xytext=((p_c) / 1000, 0.0 + y_pad),
    arrowprops=dict(
        arrowstyle="<-",
        color="black",
        lw=1.0,
    ),
    annotation_clip=False,
)

# place text within the yield surface, between p_s and p_c
mid_x = (p_c / 1000 + p_s / 1000) / 2
ax.text(
    mid_x,
    0.5 + y_pad * 1.5,
    "$2 p_\\mathrm{s}$",
    ha="center",
    va="bottom",
    color="black",
)

ax.vlines(
    x=p_s / 1000,
    ymin=0,
    ymax=M_csl * p_s / 1000,
    colors="black",
    linestyles="dashed",
    lw=1.0,
)


###########################################################################
# Configure plots
###########################################################################

# Hide ticks
ax.set_xticks([], minor=True)
ax.set_xticks([], minor=False)
ax.set_yticks([], minor=True)
ax.set_yticks([], minor=False)

# Set axis limits
ax.set_xlim(-1, 310)
ax.set_ylim(-2, 220)

# Set axis labels
ax.set_xlabel(LABELS["pressure"])
ax.set_ylabel(LABELS["deviatoric_stress"])

# Annotate axes
annotate_axes(ax, "(b)")

# Setup legend
handles, labels = sort_legend_handles([ax])
ax.legend(handles, labels, loc="upper right")

# saver figure
fig.savefig(FIGURE_DIR / "theory_yield_surface.pdf", bbox_inches="tight")
print("Saved figure to:", FIGURE_DIR / "theory_yield_surface.pdf")
print(
    "Note: You might see a warning about fixed aspect, please ignore, this is intentional."
)
