import dataclasses
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass(frozen=True)
class PlotHelper:
    x: np.ndarray
    y: np.ndarray
    xlabel: str = None
    ylabel: str = None
    xlim: List[float] = dataclasses.field(default_factory=lambda: [None, None])
    ylim: List[float] = dataclasses.field(default_factory=lambda: [None, None])

    ls: str = "-"
    marker: str = None
    color: str = None
    start_end_markers: bool = True

    xlogscale: bool = False
    ylogscale: bool = False


def make_plots(
    plot_list: List[PlotHelper],
    fig_ax: Tuple = None,
    file: str = None,
    subplots_options: Dict = None,
    savefig_options: Dict = None,
):
    import scienceplots

    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"font.size": 8})

    if subplots_options is None:
        subplots_options = {}

    subplots_options.setdefault("nrows", 2)
    subplots_options.setdefault("ncols", 3)
    subplots_options.setdefault("figsize", (8, 3))
    subplots_options.setdefault("dpi", 200)

    if fig_ax is None:
        fig, axes = plt.subplots(**subplots_options)
    else:
        fig, axes = fig_ax
    axes = axes.reshape(-1)
    for i, plot in enumerate(plot_list):
        (line,) = axes[i].plot(
            plot.x, plot.y, ls=plot.ls, marker=plot.marker, color=plot.color
        )
        axes[i].set_xlabel(plot.xlabel)
        axes[i].set_ylabel(plot.ylabel)
        axes[i].set_xlim(plot.xlim)
        axes[i].set_ylim(plot.ylim)

        if plot.start_end_markers:
            axes[i].plot(plot.x[0], plot.y[0], ".", color=line.get_color())
            axes[i].plot(plot.x[-1], plot.y[-1], "x", color=line.get_color())

        if plot.xlogscale:
            axes[i].set_xscale("log")
        if plot.ylogscale:
            axes[i].set_yscale("log")

    plt.tight_layout()
    if file is not None:
        fig.savefig(file)
        fig.clear()

    return fig, axes
