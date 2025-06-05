import os
from tkinter import font

import numpy as np

import matplotlib.pyplot as plt
import pyvista as pv

from matplotlib import cm, patches, ticker

import matplotlib as mpl
import warnings

from matplotlib.ticker import FuncFormatter


import scienceplots

plt.style.use(["science", "no-latex"])
mpl.rcParams["lines.linewidth"] = 2.5

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

dir_path = os.path.dirname(os.path.realpath(__file__))

plot_dir = os.path.join(dir_path, "plots")  # plot directory


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


def plot_contour(
    axis, x, y, hue, min_max=None, logscale=False, cmap=cm.PuBu, title=None
):
    num_vals = 21

    levels = np.linspace(min_max[0], min_max[1], num_vals)

    cp = axis.contourf(x, y, hue, levels=levels, extend="max", cmap=cmap)

    if title is not None:
        axis.set_title(title)

    return axis, cp


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


# contains simulation data
# Get simulation output files and setup time points


projects = [
    "mu_i",
    "mcc_ocr1",
    "mcc_ocr4",
]

labels = [
    "$\\mu (I)$-LC",
    "MCC OCR=1",
    "MCC OCR=4",
]

leg_label = "shear strain $\\gamma$"


grid_size = (517, 165)  # change this when adjusting orgin/end


projects_column_spread = []

projects_time_stack = []

time_stack = np.arange(0, 1.0, 0.01)


spread_to_view = [0.1, 0.5, 1.0]

fig, axes = plt.subplots(
    ncols=3,
    nrows=3,
    figsize=(8, 3.4),
    dpi=1200,
    sharex=True,
    sharey=True,
    layout="constrained",
)


for pi, project in enumerate(projects):
    sim_dir = os.path.join(dir_path, f"output/{project}")

    mp_output = get_files(sim_dir, "material_points")

    # input_arrays = np.load(sim_outputs[0])

    # grid_position_stack = input_arrays.get("position_stack", None)

    # lsit of spreads
    W_list = []

    for fi, file_ in enumerate(mp_output):
        position_stack = np.load(file_)["position_stack"]

        x_min = position_stack[:, 0].min()
        x_max = position_stack[:, 0].max()

        spread = x_max - x_min

        W_list.append(spread)

    W_list = np.array(W_list)

    W_min = W_list.min()
    W_max = W_list.max()

    W_reg = (W_list - W_min) / (W_max - W_min)

    # spread at 0.1
    # wspread1, file_id1 = find_nearest(W_reg, 0.1)
    # wspread2, file_id2 = find_nearest(W_reg, 0.3)
    # wspread3, file_id3 = find_nearest(W_reg, 1.0)

    grid_output = get_files(sim_dir, "shape_map")
    # for ai, fi in enumerate([file_id1, file_id2, file_id3]):
    for ai, spv in enumerate(spread_to_view):
        found_spv, nearest_index = find_nearest(W_reg, spv)
        input_arrays = np.load(grid_output[nearest_index])

        axes[pi, ai].set_aspect("equal")

        data = input_arrays["p2g_gamma_stack"]
        grid_position_stack = input_arrays["grid_position_stack"]
        x, y = grid_position_stack.T

        x_grid, y_grid = x.reshape(grid_size), y.reshape(grid_size)
        data_grid = data.reshape(grid_size)
        sup_title = None
        if pi == 0:
            sup_title = "$\\widetilde{w} \\approx$" + f"{spv:.2f}"

        (_, cp) = plot_contour(
            axes[pi, ai],
            x_grid,
            y_grid,
            data_grid,
            min_max=(0, 3),
            title=sup_title,
            cmap="copper",
        )
        axes[pi, ai].set_xlim(0.4, 2.8)
        axes[pi, ai].set_ylim(0.0, 1.0)

        if pi == 2:
            axes[pi, ai].set_xlabel("$x$ [m]")
            axes[pi, ai].set_xticks([0.5, 1.0, 1.5, 2.0, 2.5])
        if ai == 0:
            axes[pi, ai].set_ylabel(f"{labels[pi]}\n\n $y$ [m]")
            axes[pi, ai].set_ylabel(f"{labels[pi]}\n\n $y$ [m]")


ticks = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
cbar = fig.colorbar(
    cp,
    cax,
    format=FuncFormatter(lambda x, pos: "{:.2f}".format(x)),
    **kw,
    shrink=0.8,
    aspect=20,
    ticks=ticks,
)
cbar.ax.set_ylabel(leg_label)

plt.minorticks_off()

plt.savefig(plot_dir + "/collapse.pdf", bbox_inches="tight")
