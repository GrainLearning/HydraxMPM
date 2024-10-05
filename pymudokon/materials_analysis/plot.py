import dataclasses
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclasses.dataclass(frozen=True)
class PlotHelper:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray = None
    xlabel: str = None
    ylabel: str = None
    zlabel: str = None
    xlim: List[float] = dataclasses.field(default_factory=lambda: [None, None])
    ylim: List[float] = dataclasses.field(default_factory=lambda: [None, None])
    zlim: List[float] = dataclasses.field(default_factory=lambda: [None, None])
    
    ls: str = "-"
    markersize: int = None
    markeredgecolor: str = None
    marker: str = None
    color: str = None
    alpha: float = None
    start_end_markers: bool = True
    zorder: float = None
    xlogscale: bool = False
    ylogscale: bool = False
    label: str = None

def create_plot(plot,axis):
    (line,) = axis.plot(
        plot.x, plot.y, ls=plot.ls, marker=plot.marker, color=plot.color, label= plot.label,
        
        markersize = plot.markersize,
        markeredgecolor = plot.markeredgecolor,
        alpha = plot.alpha
    )
    if plot.start_end_markers:
        axis.plot(plot.x[0], plot.y[0], ".", color=line.get_color())
        axis.plot(plot.x[-1], plot.y[-1], "x", color=line.get_color())

    if plot.xlogscale:
        axis.set_xscale("log")
    if plot.ylogscale:
        axis.set_yscale("log")
        
    axis.set_xlabel(plot.xlabel)
    axis.set_ylabel(plot.ylabel)
    axis.set_xlim(plot.xlim)
    axis.set_ylim(plot.ylim)
    return axis
    
def make_plots(
    plot_list: List[PlotHelper]= None,
    fig_ax: Tuple = None,
    file: str = None,
    subplots_options: Dict = None,
    savefig_options: Dict = None,
    transpose_axes= False
):
    # import scienceplots

    # plt.style.use(["science", "no-latex"])
    # plt.rcParams.update({"font.size": 8})


    # # subplots_options.setdefault("dpi", 200)
    
    if subplots_options is None:
        subplots_options = {}
        
    subplots_options.setdefault("nrows", 2)
    subplots_options.setdefault("ncols", 3)
    subplots_options.setdefault("figsize", (8, 3))

    if fig_ax is None:
        fig, axes = plt.subplots(**subplots_options)
    else:
        fig, axes = fig_ax
    
    if transpose_axes:
        axes = axes.T
    
    if isinstance(axes,  (np.ndarray, np.generic) ):
        axes = axes.reshape(-1)
    
    
    if plot_list is None:
        return (fig, axes)
    

    if isinstance(axes,  (np.ndarray, np.generic) ):
        for i, plot in enumerate(plot_list):
            axes[i] = create_plot(plot,axes[i])
    else:
        axes = create_plot(plot_list[0],axes)
        
    fig.tight_layout()
    if file is not None:
        fig.savefig(file)
        fig.clear()

    return fig, axes


def add_plot(
    plot_helper: PlotHelper,
    fig_ax,
    index =0
):
    fig, axes = fig_ax
    
    if isinstance(axes,  (np.ndarray, np.generic) ):
        axes[index] = create_plot(plot_helper,axes[index])
    else:
        axes = create_plot(plot_helper,axes)
    return fig,axes



def create_plot_3d(plot,axis):
    (line,) = axis.plot(
        plot.x, plot.y, plot.z, ls=plot.ls, marker=plot.marker, color=plot.color, label= plot.label,
        
        markersize = plot.markersize,
        markeredgecolor = plot.markeredgecolor,
        alpha = plot.alpha
    )
    if plot.start_end_markers:
        axis.plot(plot.x[0], plot.y[0], plot.z[0], ".", color=line.get_color())
        axis.plot(plot.x[-1], plot.y[-1],plot.z[-1], "x", color=line.get_color())

    if plot.xlogscale:
        axis.set_xscale("log")
    if plot.ylogscale:
        axis.set_yscale("log")
        
    axis.set_xlabel(plot.xlabel)
    axis.set_ylabel(plot.ylabel)
    axis.set_zlabel(plot.zlabel)
    # axis.set_xlim(plot.xlim)
    # axis.set_ylim(plot.ylim)
    # axis.set_zlim(plot.zlim)
    return axis

def make_plots_3d(
    plot_list: List[PlotHelper]= None,
    fig_ax: Tuple = None,
    file: str = None,
    subplots_options: Dict = None,
    savefig_options: Dict = None
):

    
    if subplots_options is None:
        subplots_options = {}

    subplots_options.setdefault("nrows", 2)
    subplots_options.setdefault("ncols", 3)
    subplots_options.setdefault("figsize", (12, 8))
    subplots_options.setdefault("subplot_kw", {"projection":"3d"})

    
    if fig_ax is None:
        fig, axes = plt.subplots(**subplots_options )
    else:
        fig, axes = fig_ax
        
    if plot_list is None:
        return (fig, axes)
    

    if isinstance(axes,  (np.ndarray, np.generic) ):
        for i, plot in enumerate(plot_list):
            axes.flat[i] = create_plot_3d(plot,axes.flat[i])
    else:
        axes = create_plot_3d(plot_list[0],axes)
        
    fig.tight_layout()
    if file is not None:
        fig.savefig(file)
        fig.clear()

    return fig, axes

def add_plot_3d(
    plot_helper: PlotHelper,
    fig_ax,
    index =0
):
    fig, axes = fig_ax
    
    if isinstance(axes,  (np.ndarray, np.generic) ):
        axes.flat[index] = create_plot_3d(plot_helper,axes.flat[index])
    else:
        axes = create_plot_3d(plot_helper,axes)
    return fig,axes