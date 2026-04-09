import numpy as np
import matplotlib.pyplot as plt
import sys


import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from UEPI_MCC import (
    get_ps_I,
    get_xi,
    get_M_xi,
    get_M_I,
    get_deps_v_p,
    get_deps_s_p,
)

from plotting import (
    COLORS,
    FIG_SIZES,
    LABELS,
    FIGURE_DIR,
    annotate_axes,
    sort_legend_handles,
    annotate_point,
    plot_triangle,
    plot_orthogonality_marker,
)

###########################################################################
# Material parameters
###########################################################################

M_csl = 1.0  # critical state friction coefficient [-]
M_inf = 2.5  # dynamic friction coefficient at high inertial numbers [-]
I_M = 0.005  # charateristic inertial number for friction [-]
I_v = 0.008  # charateristic inertial number for dilation [-]

###########################################################################
# Compute State variables & state parameter
###########################################################################

I_p = 1e-3  # plastic inertial number [-]

p_s = 100_000  # shift pressure [Pa]

p_s_I = get_ps_I(p_s, I_p, I_v=I_v)

p_c = 2 * p_s  # normal consolidation pressure [Pa]

p_stack = np.linspace(1e-12, p_c * 10, 50_000)


###########################################################################
# Compute product functions
###########################################################################

xi_stack = get_xi(p_s, p_stack)
xi_I_stack = get_xi(p_s_I, p_stack)


M_xi_stack = get_M_xi(xi_stack)
M_xi_I_stack = get_M_xi(xi_I_stack)
M_I = get_M_I(I_p, I_M, M_csl, M_inf)


# M_I=v_I=1
q_ri_stack = (M_csl * M_xi_stack) * p_stack

# M_I (up)
q_ssf_stack = (M_csl * M_xi_stack * M_I) * p_stack

# v_I (up)
q_ssd_stack = (M_csl * M_xi_I_stack) * p_stack


# reference lines
q_M_iss_stack = M_csl * M_I * p_stack
q_M_csl_stack = M_csl * p_stack

###########################################################################
# Setup, plot, save
###########################################################################

fig = plt.figure(figsize=FIG_SIZES["single_col"])
fig.subplots_adjust(bottom=0.15)
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect("equal", adjustable="datalim")

ax.plot(
    p_stack / 1000,
    q_ri_stack / 1000,
    color=COLORS["ys"],
    linestyle="-",
    label=r"I-YS ($M_I=v_I=1$)",
    zorder=10,
)

ax.plot(
    p_stack / 1000,
    q_ssf_stack / 1000,
    color="#084594",
    linestyle="--",
    label=r"I-YS ($M_I \uparrow$)",
    zorder=10,
)


ax.plot(
    p_stack / 1000,
    q_ssd_stack / 1000,
    color="#a6cee3",
    linestyle=":",
    label=r"I-YS ($v_I \uparrow$)",
    zorder=10,
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
    p_stack / 1000,
    q_M_iss_stack / 1000,
    color=COLORS["csl"],
    linestyle="--",
    label="ISS",
)

# add shading
q_base_fill = np.maximum(np.nan_to_num(q_ri_stack, nan=0.0), 0.0)
q_ssf_fill = np.maximum(np.nan_to_num(q_ssf_stack, nan=0.0), 0.0)
q_ssd_fill = np.maximum(np.nan_to_num(q_ssd_stack, nan=0.0), 0.0)

# Between: I-YS (M_I=v_I=1) and I-YS (M_I up)
ax.fill_between(
    p_stack / 1000,
    q_base_fill / 1000,
    q_ssf_fill / 1000,
    facecolor="none",
    # edgecolor=COLORS["ys"],
    edgecolor="#084594",
    hatch="///",  # Forward slashes
    alpha=0.7,
    linewidth=0,
    zorder=1,
)

# Between: I-YS (M_I=v_I=1) and I-YS (v_I up)
ax.fill_between(
    p_stack / 1000,
    q_base_fill / 1000,
    q_ssd_fill / 1000,
    facecolor="none",
    edgecolor="#a6cee3",
    hatch="\\\\\\",
    alpha=0.7,
    linewidth=0,
    zorder=1,
)


### rate-independent limit ###
arrow_index = 4400
pmulti = 1000.0

flow_u = get_deps_v_p(p_stack, xi_stack,  pmulti)
flow_v = get_deps_s_p(p_stack, xi_stack, q_ri_stack, M_csl, pmulti)

ax.quiver(
    (p_stack / 1000)[arrow_index],
    (q_ri_stack / 1000)[arrow_index],
    (flow_u)[arrow_index],
    (flow_v)[arrow_index],
    scale_units="xy",
    scale=0.7 / pmulti,
    width=0.009,
    angles="xy",
    headwidth=3,
    headlength=4,
    zorder=10,
)
tan_u = (p_stack[arrow_index + 1] - p_stack[arrow_index - 1]) / 1000
tan_v = (q_ri_stack[arrow_index + 1] - q_ri_stack[arrow_index - 1]) / 1000

# orthogonality marker
plot_orthogonality_marker(
    ax,
    ((p_stack / 1000)[arrow_index], (q_ri_stack / 1000)[arrow_index]),
    ((flow_u / 1000)[arrow_index], (flow_v / 1000)[arrow_index]),
    (tan_u, tan_v),
    size=6.0,
)

# line under arrow
tangent_len = 10.0
t_vec = np.array([tan_u, tan_v])
t_vec = t_vec / np.linalg.norm(t_vec)
center_pt = np.array([(p_stack / 1000)[arrow_index], (q_ri_stack / 1000)[arrow_index]])
p_start = center_pt - (tangent_len / 2) * t_vec
p_end = center_pt + (tangent_len / 2) * t_vec
ax.plot(
    [p_start[0], p_end[0]],
    [p_start[1], p_end[1]],
    color="black",
    linestyle="-",
    lw=1,
    zorder=15,
)


### steady state dilation ###
pmulti = 1000.0

flow_u = get_deps_v_p(p_stack, xi_I_stack,  pmulti)
flow_v = get_deps_s_p(p_stack, xi_I_stack, q_ssd_stack, M_csl, pmulti)

ax.quiver(
    (p_stack / 1000)[arrow_index],
    (q_ssd_stack / 1000)[arrow_index],
    (flow_u)[arrow_index],
    (flow_v)[arrow_index],
    scale_units="xy",
    scale=0.7 / pmulti,
    width=0.009,
    angles="xy",
    headwidth=3,
    headlength=4,
    zorder=10,
)


tan_u = (p_stack[arrow_index + 1] - p_stack[arrow_index - 1]) / 1000
tan_v = (q_ssd_stack[arrow_index + 1] - q_ssd_stack[arrow_index - 1]) / 1000

plot_orthogonality_marker(
    ax,
    ((p_stack / 1000)[arrow_index], (q_ssd_stack / 1000)[arrow_index]),
    ((flow_u / 1000)[arrow_index], (flow_v / 1000)[arrow_index]),
    (tan_u, tan_v),
    size=6.0,
)
tangent_len = 10.0
t_vec = np.array([tan_u, tan_v])
t_vec = t_vec / np.linalg.norm(t_vec)
center_pt = np.array([(p_stack / 1000)[arrow_index], (q_ssd_stack / 1000)[arrow_index]])
p_start = center_pt - (tangent_len / 2) * t_vec
p_end = center_pt + (tangent_len / 2) * t_vec

ax.plot(
    [p_start[0], p_end[0]],
    [p_start[1], p_end[1]],
    color="black",
    linestyle="-",
    lw=1,
    zorder=15,
)

### steady state with inertia + bulk friction ###
pmulti = 1000.0

flow_u = get_deps_v_p(p_stack, xi_stack, pmulti)
flow_v = get_deps_s_p(p_stack, xi_stack, q_ssf_stack, M_csl, pmulti)

ax.quiver(
    (p_stack / 1000)[arrow_index],
    (q_ssf_stack / 1000)[arrow_index],
    (flow_u)[arrow_index],
    (flow_v)[arrow_index],
    scale_units="xy",
    scale=0.7 / pmulti,
    width=0.009,
    angles="xy",
    headwidth=3,
    headlength=4,
    zorder=10,
)

tan_u = (p_stack[arrow_index + 1] - p_stack[arrow_index - 1]) / 1000
tan_v = (q_ssf_stack[arrow_index + 1] - q_ssf_stack[arrow_index - 1]) / 1000

tangent_len = 10.0
t_vec = np.array([tan_u, tan_v])
t_vec = t_vec / np.linalg.norm(t_vec)
center_pt = np.array([(p_stack / 1000)[arrow_index], (q_ssf_stack / 1000)[arrow_index]])
p_start = center_pt - (tangent_len / 2) * t_vec
p_end = center_pt + (tangent_len / 2) * t_vec

ax.plot(
    [p_start[0], p_end[0]],
    [p_start[1], p_end[1]],
    color="black",
    linestyle="-",
    lw=1,
    zorder=15,
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


# Setup legend
ax.legend(loc="upper right", ncols=2, fontsize=9)


# saver figure
fig.savefig(FIGURE_DIR / "theory_inertial_yield_surface.pdf", bbox_inches="tight")
print("Saved figure to:", FIGURE_DIR / "theory_inertial_yield_surface.pdf")
print(
    "Note: You might see a warning about fixed aspect, please ignore, this is intentional."
)
