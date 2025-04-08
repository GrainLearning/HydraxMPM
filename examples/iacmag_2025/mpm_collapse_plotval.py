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

# define input and output directories
dir_path = os.path.dirname(os.path.realpath(__file__))

sim_dir = os.path.join(dir_path, "output/dp")  # contains simulation data
plot_dir = os.path.join(dir_path, "plots")  # plot directory

plt.style.use(["science", "no-latex"])
mp_color_cycle = plt.get_cmap(
    "RdYlBu", 2
)  # Two-color colormap for failure/non-failure states


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


# load experimental data from Bui, Ha H., et al. 2008
# https://doi.org/10.1002/nag.688
# data source Huo, Zenan, et al. 2025
# https://doi.org/10.1016/j.compgeo.2025.107189

failure = np.loadtxt(dir_path + "/failure.csv", delimiter=",")
surface = np.loadtxt(dir_path + "/surface.csv", delimiter=",")

# Get simulation output files and setup time points
sim_outputs = get_files(sim_dir, "material_points")
times = np.linspace(0, 1.0, len(sim_outputs))


fig, ax = plt.subplots(
    ncols=3, figsize=(8, 4), dpi=200, layout="constrained", sharey=True
)
# Plot experimental data in bottom subplot
ax.flat[2].plot(
    failure[:, 0], failure[:, 1], "-s", ms=4, color="green", label="experiment"
)
ax.flat[2].plot(surface[:, 0], surface[:, 1], "-s", ms=4, color="green")

# Plot simulation data for selected time points t= 0.0, 0.1, 1.0 s
selected_indices = [0, 10, 100]

# Get initial MP positions
position_stack0 = np.load(sim_outputs[0]).get("position_stack")


# This part is used to determine the average pressure
all_pressures = []
for i in range(100):
    input_arrays = np.load(sim_outputs[i])
    p_stack = input_arrays.get("p_stack")
    all_pressures.append(p_stack)

all_pressures = np.array(all_pressures)

all_pressures.mean()


for i, index in enumerate(selected_indices):
    # Load and process simulation data for current time step
    input_arrays = np.load(sim_outputs[index])
    position_stack = input_arrays.get("position_stack")
    displacement = position_stack - position_stack0
    is_x_gt_1 = displacement[:, 0] > 0.001  # Threshold for failure displacement

    # Set labels only for last subplot to avoid duplicate legend entries
    label_failure = None
    label_not_failure = None
    if i == 2:
        # Delta u_x  represents the horizontal displacement
        # of the material points
        label_failure = "$\\Delta u_x > 1$ mm"
        label_not_failure = "$\\Delta u_x \\leq 1$ mm"

    # Plot material points plot material points with small and large horizontal displacements
    ax.flat[i].scatter(
        position_stack[:, 0][is_x_gt_1],
        position_stack[:, 1][is_x_gt_1],
        color=mp_color_cycle(0),
        s=1,
        label=label_failure,
    )

    ax.flat[i].scatter(
        position_stack[:, 0][~is_x_gt_1],
        position_stack[:, 1][~is_x_gt_1],
        label=label_not_failure,
        color=mp_color_cycle(1),
        s=1,
    )
    # Configure subplot appearance
    ax.flat[i].set_aspect("equal")
    ax.flat[i].set_xticks(np.arange(0, 0.6, 0.1))
    ax.flat[i].set_yticks([0.1])
    ax.flat[i].set_xlim((0, 0.52))
    ax.flat[i].set_ylim((0, 0.15))
    ax.flat[i].grid(True, linestyle="--")
    ax.flat[i].set_title(f"t={times[index]:.2f} s")

    ax.flat[i].set_xlabel("x [m] ")

ax.flat[0].set_ylabel("y [m] ")  # Shared ylabel for all subplots

# Custom legend creation with adjusted marker sizes
# two reasons why custom legend is needed:
# 1. legend at the bottom of the figure of all subplots
# 2. material points in scatter plot are too small, so
#    we need to increase their size
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

# Convert material point legend to custom size
new_handles = []
for handle, label in zip(lines, labels):
    if isinstance(handle, mpl.collections.PathCollection):
        color = handle.get_facecolor()[0]
        marker_size = 6
        new_handle = Line2D(
            [], [], color=color, marker="o", linestyle="", markersize=marker_size
        )
        new_handles.append(new_handle)
    else:
        new_handles.append(handle)


fig.legend(
    new_handles,
    labels,
    ncols=3,
    loc="outside lower center",
    bbox_to_anchor=(0.5, 0.2),
    fancybox=True,
    shadow=True,
    columnspacing=0.4,
    markerscale=1,  # Disable automatic scaling since we set sizes manually
)


plt.savefig(plot_dir + "/mpm_val.png")
