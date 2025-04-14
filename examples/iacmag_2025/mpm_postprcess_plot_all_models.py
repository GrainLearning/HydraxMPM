"""

This script generates the figure of the column collapse simulation
presented in the conference paper.

"""

import os

import numpy as np

import matplotlib.pyplot as plt

import matplotlib as mpl
from matplotlib.lines import Line2D

# Using scienceplots for better styles
import scienceplots
import os
from matplotlib import cm, patches, ticker
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation

# define input and output directories
dir_path = os.path.dirname(os.path.realpath(__file__))


plot_dir = os.path.join(dir_path, "plots")  # plot directory

plt.style.use(["science", "no-latex"])

is_tranparent = True

COLOR = "black"
mpl.rcParams["text.color"] = COLOR
mpl.rcParams["axes.labelcolor"] = COLOR
mpl.rcParams["xtick.color"] = COLOR
mpl.rcParams["ytick.color"] = COLOR


def plot_contour(
    axis, x, y, hue, min_max=None, logscale=False, cmap=cm.PuBu, title=None
):
    num_vals = 20
    if min_max is not None:
        levels = np.linspace(min_max[0], min_max[1], num_vals)
    if not logscale:
        cp = axis.contourf(x, y, hue, levels=levels, extend="max", cmap=cmap)
    else:
        cp = axis.contourf(
            x,
            y,
            hue,
            extend="both",
            locator=ticker.LogLocator(),
            cmap=cmap,
        )

    xmin, xmax = axis.get_xlim()
    ymin, ymax = axis.get_ylim()

    return axis, cp


def get_files(output_dr, prefix):
    """
    Loads output files and sorts them by step number. Prefix can be `material_points` or `shape_map`.
    In this script we only use `material_points`.
    """
    all_files = [
        f for f in os.listdir(output_dr) if os.path.isfile(os.path.join(output_dr, f))
    ]

    selected_files = [x for x in all_files if prefix in x]

    selected_files = [x for x in selected_files if ".npz" in x]

    selected_files_sorted = sorted(selected_files, key=lambda x: int(x.split(".")[1]))

    return [output_dr + "/" + x for x in selected_files_sorted]


# Create figure with 3 vertical subplots
fig, ax = plt.subplots(
    nrows=4,
    ncols=1,
    figsize=(16, 4),
    dpi=300,
    layout="constrained",
    sharey=True,
    sharex=True,
)

# Plot simulation data for selected time points t= 0.0, 0.1, 1.0 s
selected_index = 100

projects = ["dp", "mcc", "mu_i", "fluid"]
models = [
    "Drucker-Prager",
    "Modified Cam-Clay",
    "$ \\mu (I)$-rheology",
    "Newtonian Fluid",
]

plot_scalar = "shear_strain"
if plot_scalar == "KE":
    leg_label = "Nodal Kinetic Energy (log-scale) [J]"
    file_suffix = "KE"
    data_name = "p2g_KE_stack"
    cmap = cm.PuBu
    logscale = True
elif plot_scalar == "shear_strain":
    cmap = "viridis"
    file_suffix = "gamma"
    data_name = "p2g_gamma_stack"
    leg_label = "Shear Strain $\gamma^l$"
    logscale = False

# Get initial MP positions
for pi, project in enumerate(projects):
    sim_dir = os.path.join(dir_path, f"output/{project}")  # contains simulation data
    sim_outputs = get_files(sim_dir, "shape_map")

    times = np.linspace(0, 1.0, len(sim_outputs))

    input_arrays = np.load(sim_outputs[selected_index])
    grid_mesh = input_arrays.get("grid_mesh")
    data_mesh = input_arrays.get(data_name).reshape(np.array(grid_mesh.shape)[[0, 1]])
    time = np.round(times[selected_index], 2)
    (_, cp) = plot_contour(
        ax[pi],
        grid_mesh[:, :, 0],
        grid_mesh[:, :, 1],
        data_mesh,
        min_max=(0, 2),
        cmap=cmap,
        logscale=logscale,
    )
    # Configure subplot appearance
    # ax.flat[i].set_xticks(np.arange(0, 0.6, 0.1))
    ax[pi].set_ylim((0, 0.15))
    ax[pi].set_aspect("equal")
    # ax[i, pi].grid(True, linestyle="--")
    ax[pi].xaxis.set_ticklabels([])
    ax[pi].yaxis.set_ticklabels([])

    ax[pi].set_aspect("equal")

    ax[pi].text(
        0.4,
        0.9,
        models[pi],
        fontsize=9,
        ha="left",
        va="top",
        transform=ax[pi].transAxes,
        bbox=dict(alpha=0.0),
    )
    ax[pi].tick_params(axis="both", which="both", length=0)
    ax[pi].margins(0.0)

# Calculate colorbar position based on subplot dimensions
pos = ax[3].get_position()
cax = fig.add_axes(
    [
        pos.x1 + 0.025,  # Start right of last subplot
        pos.y0,  # Align with bottom of first subplot
        0.015,  # Width of colorbar
        pos.height * 5,  # Height spans all subplots
    ]
)

# Create colorbar
if logscale:
    cbar = fig.colorbar(
        cp,
        cax=cax,
        format=LogFormatterSciNotation(),
    )
else:
    fmt = lambda x, pos: "{:.2f}".format(x)

    cbar = fig.colorbar(
        cp,
        cax=cax,
        format=FuncFormatter(fmt),
    )
cbar.ax.set_ylabel(leg_label)


fig.suptitle('Column Collapse (t=1.0 s)', fontsize=8)
plt.savefig(
    plot_dir + f"/mpm_plot_contour_{file_suffix}.png", transparent=is_tranparent
)
